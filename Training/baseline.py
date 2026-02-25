import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, classification_report
from torch.optim import AdamW
import os
import random

# ==========================================
# 配置与常量
# ==========================================
class Config:
    def __init__(self):
        self.data_dir = "cache"
        self.batch_size = 64
        self.epochs = 10
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.hidden_dim = 256
        self.num_labels = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_runs = 5

# ==========================================
# Step 1: 高效数据集加载 (支持多种模式)
# ==========================================
class FactCheckDataset(Dataset):
    def __init__(self, data_dir="cache", mode="claim_only"):
        """
        mode: 'claim_only' | 'claim_evidence'
        """
        self.mode = mode
        self.data_dir = data_dir
        
        # 1. 加载 Claim 嵌入 (必须)
        claim_path = f"{data_dir}/claims_embeddings.npy"
        if os.path.exists(claim_path):
            self.claims = np.load(claim_path)
        else:
            raise FileNotFoundError(f"缺少核心文件：{claim_path}，请先生成 Claim 嵌入。")
            
        # 2. 加载 Evidence 嵌入 (可选，取决于模式)
        self.evidences = None
        self.evd_masks = None
        if mode == "claim_evidence":
            evd_path = f"{data_dir}/evidences_embeddings_prev.npy"
            mask_path = f"{data_dir}/evd_mask.npy"
            if os.path.exists(evd_path) and os.path.exists(mask_path):
                self.evidences = np.load(evd_path)
                self.evd_masks = np.load(mask_path)
            else:
                raise FileNotFoundError(f"缺少证据文件：{evd_path} 或 {mask_path}")

        # 3. 加载标签
        label_path = f"{data_dir}/labels.npy"
        if os.path.exists(label_path):
            self.labels = np.load(label_path)
        else:
            raise FileNotFoundError(f"缺少标签文件：{label_path}")

        print(f"[Dataset] Loaded {len(self.labels)} samples. Mode: {mode}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "claim_emb": torch.from_numpy(self.claims[idx]).float(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
        if self.mode == "claim_evidence" and self.evidences is not None:
            item["evidence_emb"] = torch.from_numpy(self.evidences[idx]).float()
            item["evidence_mask"] = torch.from_numpy(self.evd_masks[idx]).float()
            
        return item

# ==========================================
# Step 2: 灵活分类模型 (支持拼接)
# ==========================================
class FactCheckClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_labels=3):
        super().__init__()
        # 动态输入维度：Claim (768) 或 Claim+Evidence (768+768)
        self.input_dim = input_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_labels)
        )

    def forward(self, claim_emb, evidence_emb=None, evidence_mask=None):
        """
        claim_emb: (batch, 768)
        evidence_emb: (batch, 5, 768) [Optional]
        evidence_mask: (batch, 5) [Optional]
        """
        if evidence_emb is not None and evidence_mask is not None:
            # 1. 证据掩码平均池化
            mask_unsqueezed = evidence_mask.unsqueeze(-1)
            sum_embeddings = torch.sum(evidence_emb * mask_unsqueezed, dim=1)
            counts = torch.clamp(torch.sum(mask_unsqueezed, dim=1), min=1e-9)
            pooled_evidence = sum_embeddings / counts
            
            # 2. 拼接 Claim 和 Evidence 特征
            # 核心逻辑：事实核查需要 Claim 与 Evidence 的交互
            combined_rep = torch.cat([claim_emb, pooled_evidence], dim=1)
        else:
            # 仅使用 Claim 特征
            combined_rep = claim_emb
            
        logits = self.mlp(combined_rep)
        return logits

# ==========================================
# Step 3: 训练与评估核心逻辑
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        claim_emb = batch["claim_emb"].to(device)
        labels = batch["label"].to(device)
        
        evidence_emb = batch.get("evidence_emb", None)
        evidence_mask = batch.get("evidence_mask", None)
        if evidence_emb is not None:
            evidence_emb = evidence_emb.to(device)
            evidence_mask = evidence_mask.to(device)

        optimizer.zero_grad()
        logits = model(claim_emb, evidence_emb, evidence_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_golds = []
    
    with torch.no_grad():
        for batch in loader:
            claim_emb = batch["claim_emb"].to(device)
            labels = batch["label"].to(device)
            
            evidence_emb = batch.get("evidence_emb", None)
            evidence_mask = batch.get("evidence_mask", None)
            if evidence_emb is not None:
                evidence_emb = evidence_emb.to(device)
                evidence_mask = evidence_mask.to(device)

            logits = model(claim_emb, evidence_emb, evidence_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_golds.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_golds

def run_experiment(mode, config, seed):
    """运行单次实验，返回指标"""
    # 设置随机种子以保证可复现性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # 1. 数据准备
    dataset = FactCheckDataset(data_dir=config.data_dir, mode=mode)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    
    # 2. 模型初始化
    # 如果是 claim_only, input=768; 如果是 claim_evidence, input=768+768=1536
    input_dim = 768 if mode == "claim_only" else 1536
    model = FactCheckClassifier(input_dim=input_dim, hidden_dim=config.hidden_dim).to(config.device)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # 3. 训练循环
    best_f1 = 0
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.device)
        val_loss, preds, golds = evaluate(model, val_loader, criterion, config.device)
        
        # 计算验证集 F1
        macro_f1 = f1_score(golds, preds, average='macro')
        micro_f1 = f1_score(golds, preds, average='micro')
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            # 这里可以添加保存最佳模型的逻辑
            
    # 返回最后一次评估结果（或最佳结果，此处简化为最后一次）
    # 为了严谨，我们重新在验证集上评估一次最佳状态（简化版直接返回最后一次）
    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "val_loss": val_loss
    }

# ==========================================
# Step 4: 主执行流程与结果汇总
# ==========================================
def main():
    config = Config()
    print(f"=== 开始事实核查基线实验 (Device: {config.device}) ===")
    
    experiments = [
        {"name": "1. Claim Only (BERT Embeddings)", "mode": "claim_only"},
        {"name": "2. Claim + Evidence (Fusion)", "mode": "claim_evidence"}
    ]
    
    results_summary = []
    
    for exp in experiments:
        print(f"\n>>> 正在运行实验：{exp['name']}")
        run_metrics = []
        
        for i in range(config.num_runs):
            seed = 42 + i
            print(f"   -> Run {i+1}/{config.num_runs} (Seed: {seed})...")
            try:
                metrics = run_experiment(exp["mode"], config, seed)
                run_metrics.append(metrics)
            except FileNotFoundError as e:
                print(f"   -> 错误：{e}")
                print("   -> 跳过该实验，请检查 cache 目录文件。")
                run_metrics = []
                break
        
        if run_metrics:
            # 计算均值和标准差
            micro_mean = np.mean([m["micro_f1"] for m in run_metrics])
            micro_std = np.std([m["micro_f1"] for m in run_metrics])
            macro_mean = np.mean([m["macro_f1"] for m in run_metrics])
            macro_std = np.std([m["macro_f1"] for m in run_metrics])
            
            results_summary.append({
                "Experiment": exp["name"],
                "Micro-F1": f"{micro_mean:.4f} ± {micro_std:.4f}",
                "Macro-F1": f"{macro_mean:.4f} ± {macro_std:.4f}",
                "Runs": config.num_runs
            })
            
            print(f"   -> 完成。Macro-F1: {macro_mean:.4f} (+-{macro_std:.4f})")
        else:
            results_summary.append({
                "Experiment": exp["name"],
                "Micro-F1": "N/A (File Missing)",
                "Macro-F1": "N/A (File Missing)",
                "Runs": 0
            })

    # 打印汇总表格
    print("\n" + "="*60)
    print("=== 实验结果汇总分布表 ===")
    print("="*60)
    print(f"{'Experiment':<35} | {'Micro-F1':<20} | {'Macro-F1':<20}")
    print("-" * 60)
    for res in results_summary:
        print(f"{res['Experiment']:<35} | {res['Micro-F1']:<20} | {res['Macro-F1']:<20}")
    print("="*60)
    
    # 打印最后一次运行的详细分类报告
    if len(results_summary) > 0 and results_summary[-1]['Runs'] > 0:
        print("\n[Detail] 最后一次运行 (Claim+Evidence) 分类报告:")
        # 这里为了简洁不再重新跑一次 evaluate，实际使用可保存 best model 后重新预测
        print("(详见训练过程中的输出或添加保存模型后重新加载推理的代码)")

if __name__ == "__main__":
    main()