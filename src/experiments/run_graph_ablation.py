"""
Ablation runner for graph-based evidence reasoning.

Experiments:
  - no_tfidf: disable TF-IDF weight input
  - no_ce:    disable claim-evidence (CE) features
  - no_ee:    disable evidence-evidence (EE) features
  - no_feats: disable all evidence features (CE+EE) and TF-IDF
"""
import os
import sys
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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

# Project root for direct execution: python src/experiments/run_graph_ablation.py
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.graph_dataset import GraphFactCheckDataset, graph_collate_fn
from src.models.graph_evidence_model import GraphEvidenceReasoner, GraphBatch, build_graph_batch_from_sample


@dataclass
class AblationConfig:
    name: str
    use_tfidf: bool
    feature_mask: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _basic_metrics(all_labels: np.ndarray, all_preds: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    try:
        mcc = matthews_corrcoef(all_labels, all_preds)
    except Exception:
        mcc = 0.0
    return {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "mcc": float(mcc),
    }


def apply_feature_mask(dataset: GraphFactCheckDataset, mask: np.ndarray) -> None:
    if mask is None:
        return
    if mask.ndim != 1 or mask.shape[0] != dataset.evd_feats.shape[2]:
        raise ValueError(
            f"feature_mask shape mismatch: got {mask.shape}, expected ({dataset.evd_feats.shape[2]},)"
        )
    dataset.evd_feats = dataset.evd_feats * mask[None, None, :]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []
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

        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        labs = labels.detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labs)

    all_preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    all_labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    metrics = (
        _basic_metrics(all_labels, all_preds)
        if all_labels.size
        else {"acc": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "mcc": 0.0}
    )
    return total_loss / max(n_batches, 1), metrics


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

    base = _basic_metrics(all_labels, all_preds)
    acc = base["acc"]
    macro_f1 = base["macro_f1"]
    micro_f1 = base["micro_f1"]
    mcc = base["mcc"]

    # Real / Fake 的 Precision / Recall / F1
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


def _write_json(path: Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, payload: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_experiment(
    config: AblationConfig,
    seed: int,
    cache_dir: str,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    epochs: int,
    lr: float,
) -> Dict[str, float]:
    set_seed(seed)

    dataset = GraphFactCheckDataset(cache_dir=cache_dir, use_tfidf=config.use_tfidf, device=None)
    apply_feature_mask(dataset, config.feature_mask)

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=g)

    # 类别不平衡的加权损失
    label_array = dataset.labels.astype(int)
    class_counts = np.bincount(label_array)
    num_labels = len(class_counts)
    total = class_counts.sum()
    class_weights_np = total / (np.maximum(class_counts, 1) * num_labels)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=graph_collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=graph_collate_fn,
    )

    node_dim = dataset.claim_embs.shape[1] + dataset.evd_feats.shape[2] + (1 if dataset.tfidf is not None else 0)
    edge_dim = 5
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

    best_loss = float("inf")
    best_metrics = {}

    epoch_log_path = out_dir / "epoch_log.jsonl"
    if epoch_log_path.exists():
        epoch_log_path.unlink()

    for epoch in range(epochs):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device)
        print(
            f"[{config.name} seed={seed}] Epoch {epoch+1}/{epochs}  "
            f"loss={train_loss:.4f}  acc={metrics['acc']:.4f}  "
            f"macro_f1={metrics['macro_f1']:.4f}  mcc={metrics['mcc']:.4f}"
        )
        _write_jsonl(
            epoch_log_path,
            {
                "exp_name": config.name,
                "seed": seed,
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_acc": train_metrics["acc"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_mcc": train_metrics["mcc"],
                "val_acc": metrics["acc"],
                "val_macro_f1": metrics["macro_f1"],
                "val_micro_f1": metrics["micro_f1"],
                "val_mcc": metrics["mcc"],
            },
        )
        if train_loss < best_loss:
            best_loss = float(train_loss)
            best_metrics = dict(metrics)
            best_metrics["best_epoch"] = epoch + 1
            best_metrics["best_loss"] = best_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics},
                out_dir / "best_model.pt",
            )

    _write_json(
        out_dir / "metrics.json",
        {
            "exp_name": config.name,
            "seed": seed,
            "best": best_metrics,
        },
    )
    return best_metrics


def _avg_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    def mean(key: str) -> float:
        vals = [m.get(key, 0.0) for m in metrics_list]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "macro_f1_avg": mean("macro_f1"),
        "micro_f1_avg": mean("micro_f1"),
        "real_precision_avg": mean("real_precision"),
        "real_recall_avg": mean("real_recall"),
        "real_f1_avg": mean("real_f1"),
        "fake_precision_avg": mean("fake_precision"),
        "fake_recall_avg": mean("fake_recall"),
        "fake_f1_avg": mean("fake_f1"),
        "mcc_avg": mean("mcc"),
    }


def main():
    cache_dir = os.environ.get("CACHE_DIR", "cache")
    root_out_dir = Path(os.environ.get("OUT_DIR", "outputs/graph/ablation"))
    root_out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    epochs = int(os.environ.get("EPOCHS", "20"))
    lr = float(os.environ.get("LR", "1e-4"))
    seed = int(os.environ.get("SEED", "42"))

    d_feat = 5
    full_mask = np.ones(d_feat, dtype=np.float32)
    no_ce_mask = np.array([1, 0, 0, 1, 1], dtype=np.float32)
    no_ee_mask = np.array([1, 1, 1, 0, 0], dtype=np.float32)
    no_feat_mask = np.zeros(d_feat, dtype=np.float32)

    experiments: List[AblationConfig] = [
        AblationConfig(name="no_tfidf", use_tfidf=False, feature_mask=full_mask),
        AblationConfig(name="no_ce", use_tfidf=True, feature_mask=no_ce_mask),
        AblationConfig(name="no_ee", use_tfidf=True, feature_mask=no_ee_mask),
        AblationConfig(name="no_feats", use_tfidf=False, feature_mask=no_feat_mask),
    ]

    for config in experiments:
        exp_out_dir = root_out_dir / config.name / f"seed_{seed}"
        exp_out_dir.mkdir(parents=True, exist_ok=True)
        best_metrics = run_experiment(
            config=config,
            seed=seed,
            cache_dir=cache_dir,
            out_dir=exp_out_dir,
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
        )

        _write_json(
            exp_out_dir / "summary.json",
            {
                "exp_name": config.name,
                "seed": seed,
                "macro_f1": best_metrics.get("macro_f1", 0.0),
                "micro_f1": best_metrics.get("micro_f1", 0.0),
                "real_precision": best_metrics.get("real_precision", 0.0),
                "real_recall": best_metrics.get("real_recall", 0.0),
                "real_f1": best_metrics.get("real_f1", 0.0),
                "fake_precision": best_metrics.get("fake_precision", 0.0),
                "fake_recall": best_metrics.get("fake_recall", 0.0),
                "fake_f1": best_metrics.get("fake_f1", 0.0),
                "mcc": best_metrics.get("mcc", 0.0),
            },
        )

        avg_metrics = _avg_metrics([best_metrics])
        _write_json(
            root_out_dir / config.name / "summary_avg.json",
            {"exp_name": config.name, **avg_metrics},
        )


if __name__ == "__main__":
    main()
