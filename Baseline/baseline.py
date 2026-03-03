import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
from torch.optim import AdamW

# -----------------------------------------
# Step 1: 高效 Npy 数据集加载
# -----------------------------------------
class NpyFactCheckDataset(Dataset):
    def __init__(self, data_dir="cache"):
        # 加载你生成的三个核心特征文件
        
        # After evidences optimized log 
        self.embeddings = np.load(f"{data_dir}/evidences_embeddings_r.npy") # (N, 5, 768)
        self.masks = np.load(f"{data_dir}/evd_mask_r.npy")                 # (N, 5)
        
        # self.embeddings = np.load(f"{data_dir}/evidences_embeddings_prev.npy") # (N, 5, 768)
        # self.masks = np.load(f"{data_dir}/evd_mask.npy")                 # (N, 5)
        self.labels = np.load(f"{data_dir}/labels.npy")                           # (N,)
        
        print(f"[Sigma] Loaded {len(self.labels)} samples from {data_dir}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embeddings": torch.from_numpy(self.embeddings[idx]).float(),
            "mask": torch.from_numpy(self.masks[idx]).float(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -----------------------------------------
# Step 2: Sigma 掩码分类模型
# -----------------------------------------
class SigmaVectorClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_labels=3):
        super().__init__()
        # 深度特征提取网络
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_labels)
        )

    def forward(self, embeddings, mask):
        """
        embeddings: (batch, 5, 768)
        mask: (batch, 5)
        """
        # 1. 掩码平均池化 (Masked Mean Pooling)
        # 目的：忽略掉那些全零（空内容）的证据向量
        mask_unsqueezed = mask.unsqueeze(-1) # (batch, 5, 1)
        sum_embeddings = torch.sum(embeddings * mask_unsqueezed, dim=1) # (batch, 768)
        counts = torch.clamp(torch.sum(mask_unsqueezed, dim=1), min=1e-9)
        
        pooled_rep = sum_embeddings / counts # 得到聚合后的证据表示
        
        # 2. 分类推理
        logits = self.mlp(pooled_rep)
        return logits

# -----------------------------------------
# Step 3: 训练与评估核心
# -----------------------------------------
def train_and_eval():
    # 配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    CACHE_DIR = "cache"
    NUM_RUNS = 5 

    dataset = NpyFactCheckDataset(CACHE_DIR)
    
    # 用于存储 5 次运行的所有指标字典
    all_reports = []
    all_runs_macro_f1 = []
    all_runs_micro_f1 = []

    print(f"Starting {NUM_RUNS} independent training runs on {DEVICE}...")

    for run in range(NUM_RUNS):
        # 重新划分与初始化
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = SigmaVectorClassifier().to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # 训练（静默模式）
        for epoch in range(EPOCHS):
            model.train()
            for batch in train_loader:
                emb = batch["embeddings"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                optimizer.zero_grad()
                logits = model(emb, mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        # 验证
        model.eval()
        all_preds, all_golds = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["embeddings"].to(DEVICE), batch["mask"].to(DEVICE))
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_golds.extend(batch["label"].numpy())

        # 核心：获取结构化字典格式的报告
        report = classification_report(all_golds, all_preds, 
                                       target_names=['Real', 'Fake'], output_dict=True)
        all_reports.append(report)
        
        # 记录 F1 分布
        all_runs_macro_f1.append(f1_score(all_golds, all_preds, average='macro'))
        all_runs_micro_f1.append(f1_score(all_golds, all_preds, average='micro'))
        
        print(f"Run {run+1}/{NUM_RUNS} completed.")

    # --- 聚合分布计算 ---
    def get_stat(key_path, metric):
        """ 从多份报告中提取特定指标的列表 """
        return [r[key_path][metric] for r in all_reports]

    def format_cell(vals):
        """ 格式化为 mean ± std """
        return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"

    # --- 最终打印输出 ---
    print("\n" + "="*60)
    print(f"[Sigma] Final Distribution Report ({NUM_RUNS} Runs)")
    print("="*60)
    
    # 打印全局 F1 分布
    print(f"Macro-F1 Avg: {format_cell(all_runs_macro_f1)}")
    print(f"Micro-F1 Avg: {format_cell(all_runs_micro_f1)}")
    print("-" * 60)

    # 打印分布表格样式
    headers = ["Category", "Precision", "Recall", "F1-score"]
    print(f"{headers[0]:<15} {headers[1]:<20} {headers[2]:<20} {headers[3]:<20}")
    
    for label in ['Real', 'Fake', 'macro avg', 'weighted avg']:
        p = get_stat(label, 'precision')
        r = get_stat(label, 'recall')
        f = get_stat(label, 'f1-score')
        print(f"{label:<15} {format_cell(p):<20} {format_cell(r):<20} {format_cell(f):<20}")

    # 准确率单独处理（因为结构不同）
    acc_list = [r['accuracy'] for r in all_reports]
    print(f"{'accuracy':<15} {' ':<20} {' ':<20} {format_cell(acc_list):<20}")
    print("="*60)

if __name__ == "__main__":
    train_and_eval()