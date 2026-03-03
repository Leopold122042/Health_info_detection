"""
图证据推理网络训练脚本。

- 从 cache 加载图数据集（GraphFactCheckDataset），使用 graph_collate_fn 返回 list[GraphSample]，
  避免不同图节点/边数量不同导致的 stack 报错。
- 每个 batch 内逐样本前向，再合并 logits/labels 计算 loss。
"""
import os
import sys
import json
import random
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    classification_report,
)

# 项目根目录加入 path，便于直接运行 python src/train_graph.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.graph_dataset import GraphFactCheckDataset, graph_collate_fn
from src.models.graph_evidence_model import GraphEvidenceReasoner, GraphBatch, build_graph_batch_from_sample


def get_dims_from_dataset(dataset: GraphFactCheckDataset):
    """从数据集推断 node_dim / edge_dim，避免依赖首样本边数。"""
    d_emb = dataset.claim_embs.shape[1]
    d_feat = dataset.evd_feats.shape[2]
    use_tf = dataset.tfidf is not None
    node_dim = d_emb + d_feat + (1 if use_tf else 0)
    edge_dim = 5  # 与 graph_dataset 中 EDGE_DIM 一致
    return node_dim, edge_dim


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        logits_list = []
        labels_list = []
        for sample in batch:
            gb = build_graph_batch_from_sample(sample)
            gb = GraphBatch(
                node_feats=gb.node_feats.to(device),
                edge_index=gb.edge_index.to(device),
                edge_feats=gb.edge_feats.to(device),
            )
            logits = model(gb)
            logits_list.append(logits if logits.dim() > 1 else logits.unsqueeze(0))
            labels_list.append(sample.label.to(device))
        logits = torch.cat(logits_list, dim=0)
        labels = torch.stack(labels_list, dim=0)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for batch in loader:
        for sample in batch:
            gb = build_graph_batch_from_sample(sample)
            gb = GraphBatch(
                node_feats=gb.node_feats.to(device),
                edge_index=gb.edge_index.to(device),
                edge_feats=gb.edge_feats.to(device),
            )
            logits = model(gb)
            pred = logits.argmax(dim=-1).squeeze().cpu().item()
            all_preds.append(pred)
            all_labels.append(sample.label.cpu().item())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)

    # 详细分类报告：Real / Fake 的 Precision / Recall / F1
    try:
        cls_report = classification_report(
            all_labels,
            all_preds,
            labels=[0, 1],
            target_names=["Real", "Fake"],
            output_dict=True,
            zero_division=0,
        )
        real_metrics = cls_report.get("Real", {})
        fake_metrics = cls_report.get("Fake", {})
        real_p = real_metrics.get("precision", 0.0)
        real_r = real_metrics.get("recall", 0.0)
        real_f1 = real_metrics.get("f1-score", 0.0)
        fake_p = fake_metrics.get("precision", 0.0)
        fake_r = fake_metrics.get("recall", 0.0)
        fake_f1 = fake_metrics.get("f1-score", 0.0)
    except Exception:
        cls_report = {}
        real_p = real_r = real_f1 = 0.0
        fake_p = fake_r = fake_f1 = 0.0
    try:
        mcc = matthews_corrcoef(all_labels, all_preds)
    except Exception:
        mcc = 0.0

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "mcc": mcc,
        "real_precision": real_p,
        "real_recall": real_r,
        "real_f1": real_f1,
        "fake_precision": fake_p,
        "fake_recall": fake_r,
        "fake_f1": fake_f1,
        "classification_report": cls_report,
    }


def main():
    seed = int(os.environ.get("SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cache_dir = os.environ.get("CACHE_DIR", "cache")
    out_dir = Path(os.environ.get("OUT_DIR", "outputs/graph"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    epochs = 20
    lr = 1e-4

    dataset = GraphFactCheckDataset(cache_dir=cache_dir, use_tfidf=True, device=None)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=g)

    # 统计类别分布，用于类别不平衡的加权损失
    label_array = dataset.labels.astype(int)
    class_counts = np.bincount(label_array)
    num_labels = len(class_counts)
    total = class_counts.sum()
    # 反比例权重：样本少的类别权重大
    class_weights_np = total / (np.maximum(class_counts, 1) * num_labels)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=graph_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=graph_collate_fn,
    )

    node_dim, edge_dim = get_dims_from_dataset(dataset)
    model = GraphEvidenceReasoner(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=256,
        num_layers=2,
        num_labels=num_labels,
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_mcc = -1.0
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch+1}/{epochs}  loss={train_loss:.4f}  "
            f"acc={metrics['acc']:.4f}  macro_f1={metrics['macro_f1']:.4f}  mcc={metrics['mcc']:.4f}"
        )
        if metrics["mcc"] > best_mcc:
            best_mcc = metrics["mcc"]
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics},
                out_dir / "best_model.pt",
            )

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved best model and metrics to {out_dir}")


if __name__ == "__main__":
    main()
