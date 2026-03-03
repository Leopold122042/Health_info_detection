import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    classification_report, matthews_corrcoef, roc_auc_score
)
from torch.utils.data import Subset, DataLoader
from utils.data_loader import HealthGraphDataset
from models.sca_gnn import SCAGNN

def get_class_weights(labels):
    """根据标签分布自动计算类别权重，平衡不平衡数据"""
    count_0 = np.sum(labels == 0)
    count_1 = np.sum(labels == 1)
    total = len(labels)
    
    # 权重公式: w = total / (num_classes * count)
    w0 = total / (2 * count_0) if count_0 > 0 else 1.0
    w1 = total / (2 * count_1) if count_1 > 0 else 1.0
    
    print(f"数据分布统计: True(0): {count_0} ({count_0/total:.2%}), Fake(1): {count_1} ({count_1/total:.2%})")
    print(f"自动计算权重: True: {w0:.4f}, Fake: {w1:.4f}")
    return torch.tensor([w0, w1], dtype=torch.float32)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits, _ = model(batch)
        loss = criterion(logits, batch['label'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_and_export(model, loader, device, fold):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    export_data = []

    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            logits, attn_dist = model(batch_gpu)
            
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            labels = batch['label'].numpy()
            attn = attn_dist.squeeze(-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs[:, 1]) # 用于计算 AUC

            for i in range(len(preds)):
                export_data.append({
                    "fold": fold,
                    "label": int(labels[i]),
                    "prediction": int(preds[i]),
                    "attention_weights": attn[i].tolist()
                })

    metrics = {
        "acc": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average='macro', zero_division=0),
        "f1_micro": f1_score(all_labels, all_preds, average='micro', zero_division=0),
        "mcc": matthews_corrcoef(all_labels, all_preds),
        "auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
        "report_dict": classification_report(all_labels, all_preds, target_names=['True', 'Fake'], output_dict=True, zero_division=0)
    }
    return metrics, export_data

def run_kfold(k=5, epochs=20, batch_size=16, lr=1e-4, model_params=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HealthGraphDataset()
    
    # 任务2: 开始前统计并引入权重
    class_weights = get_class_weights(dataset.labels).to(device)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results_dir = Path("outputs/results")
    ckpt_dir = Path("outputs/checkpoints")
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    final_export_json = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        print(f"\n>>> Fold {fold} Training Started")
        
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

        # 将消融实验参数传递给模型
        if model_params is None:
            model_params = {"use_nli": True, "use_tfidf": True, "use_ee": True}
            
        model = SCAGNN(**model_params).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_f1 = 0
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics, _ = evaluate_and_export(model, test_loader, device, fold)
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(model.state_dict(), ckpt_dir / f"best_model_fold_{fold}.pt")

        # 加载最优模型进行最终统计
        model.load_state_dict(torch.load(ckpt_dir / f"best_model_fold_{fold}.pt"))
        final_metrics, fold_export = evaluate_and_export(model, test_loader, device, fold)
        fold_metrics.append(final_metrics)
        final_export_json.extend(fold_export)

    # 任务1: 格式化打印结果并计算波动范围
    print_final_report(fold_metrics)
    # 返回平均指标供消融脚本汇总
    return {
        "acc": np.mean([m['acc'] for m in fold_metrics]),
        "f1": np.mean([m['f1'] for m in fold_metrics]),
        "mcc": np.mean([m['mcc'] for m in fold_metrics])
    }
    
    # 保存导出数据
    with open(results_dir / "model_predictions.json", "w", encoding="utf-8") as f:
        json.dump(final_export_json, f, indent=4, ensure_ascii=False)

def print_final_report(metrics_list):
    accs = [m['acc'] for m in metrics_list]
    f1s = [m['f1'] for m in metrics_list]
    micros = [m['f1_micro'] for m in metrics_list]
    mccs = [m['mcc'] for m in metrics_list]
    aucs = [m['auc'] for m in metrics_list]
    
    # 提取各折中各类的明细数据
    classes = ['True', 'Fake']
    class_stats = {cls: {'p': [], 'r': [], 'f1': []} for cls in classes}
    for m in metrics_list:
        for cls in classes:
            class_stats[cls]['p'].append(m['report_dict'][cls]['precision'])
            class_stats[cls]['r'].append(m['report_dict'][cls]['recall'])
            class_stats[cls]['f1'].append(m['report_dict'][cls]['f1-score'])

    print("\n" + "="*25 + " 全局性能统计 (5-Fold) " + "="*25)
    print(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Macro-F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Micro-F1: {np.mean(micros):.4f} ± {np.std(micros):.4f}")
    print(f"  MCC:      {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")
    print(f"  AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print("\n" + "="*23 + " 类别级指标波动分析 " + "="*23)
    print(f"{'Class':<10} | {'Precision':<15} | {'Recall':<15} | {'F1-score':<15}")
    print("-" * 70)
    for cls in classes:
        p_mean, p_std = np.mean(class_stats[cls]['p']), np.std(class_stats[cls]['p'])
        r_mean, r_std = np.mean(class_stats[cls]['r']), np.std(class_stats[cls]['r'])
        f_mean, f_std = np.mean(class_stats[cls]['f1']), np.std(class_stats[cls]['f1'])
        print(f"{cls:<10} | {p_mean:.4f}±{p_std:.3f} | {r_mean:.4f}±{r_std:.3f} | {f_mean:.4f}±{f_std:.3f}")
    print("="*70)

if __name__ == "__main__":
    run_kfold()