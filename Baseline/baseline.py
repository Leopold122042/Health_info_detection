import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


class NpyFactCheckDataset(Dataset):
    """Load cached numpy arrays for claim-evidence classification."""

    def __init__(self, data_dir: str = "cache", variant: str = "prev"):
        if variant not in {"prev", "r"}:
            raise ValueError("variant must be one of {'prev', 'r'}")

        emb_file = "evidences_embeddings_prev.npy" if variant == "prev" else "evidences_embeddings_r.npy"
        mask_file = "evd_mask_prev.npy" if variant == "prev" else "evd_mask_r.npy"

        self.embeddings = np.load(os.path.join(data_dir, emb_file))
        self.masks = np.load(os.path.join(data_dir, mask_file))
        self.labels = np.load(os.path.join(data_dir, "labels.npy")).astype(np.int64)

        claim_path = os.path.join(data_dir, "claims_embeddings.npy")
        if os.path.exists(claim_path):
            self.claims = np.load(claim_path)
        else:
            # Fall back to masked evidence average if claim embeddings are unavailable.
            mask = self.masks[:, :, None]
            summed = (self.embeddings * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
            self.claims = summed / counts

        if not (
            len(self.embeddings)
            == len(self.masks)
            == len(self.labels)
            == len(self.claims)
        ):
            raise ValueError("inconsistent sample count across cached arrays")

        self.variant = variant
        self.num_labels = int(self.labels.max()) + 1
        self.embedding_dim = int(self.embeddings.shape[-1])

        print(f"[BaselineNew] Loaded {len(self.labels)} samples from {data_dir} (variant={variant})")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "embeddings": torch.from_numpy(self.embeddings[idx]).float(),
            "mask": torch.from_numpy(self.masks[idx]).float(),
            "claim": torch.from_numpy(self.claims[idx]).float(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    """Softmax with binary mask support and stable handling of all-masked rows."""
    mask = mask.float()
    masked_scores = scores.masked_fill(mask == 0, -1e9)
    max_score = torch.max(masked_scores, dim=dim, keepdim=True).values
    probs = torch.exp(masked_scores - max_score) * mask
    denom = probs.sum(dim=dim, keepdim=True).clamp(min=eps)
    return probs / denom


class TextCNNClassifier(nn.Module):
    """TextCNN over evidence slots (sequence length=5)."""

    def __init__(
        self,
        input_dim: int = 768,
        num_labels: int = 2,
        num_filters: int = 128,
        kernel_sizes: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        kernel_sizes = kernel_sizes or [2, 3, 4, 5]
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor, claim: torch.Tensor = None) -> torch.Tensor:
        x = embeddings * mask.unsqueeze(-1)
        x = x.transpose(1, 2)

        conv_feats = []
        for conv in self.convs:
            feat = torch.relu(conv(x))
            pooled = torch.max(feat, dim=-1).values
            conv_feats.append(pooled)

        fused = torch.cat(conv_feats, dim=-1)
        fused = self.dropout(fused)
        return self.classifier(fused)


class InducTGCNClassifier(nn.Module):
    """A compact inductive GCN-style classifier for claim-evidence graphs."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gcn_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    @staticmethod
    def _build_adj(nodes: torch.Tensor) -> torch.Tensor:
        """
        Build normalized adjacency:
        - claim node (index 0) connected to all evidence nodes
        - evidence-evidence edges weighted by positive cosine similarity
        """
        n = nodes.size(0)
        device = nodes.device
        adj = torch.eye(n, device=device)

        if n > 1:
            adj[0, 1:] = 1.0
            adj[1:, 0] = 1.0

        if n > 2:
            evidence_nodes = F.normalize(nodes[1:], dim=-1)
            sim = torch.matmul(evidence_nodes, evidence_nodes.transpose(0, 1))
            sim = torch.relu(sim)
            sim.fill_diagonal_(0.0)
            adj[1:, 1:] = adj[1:, 1:] + sim

        degree = adj.sum(dim=-1).clamp(min=1e-9)
        inv_sqrt = degree.pow(-0.5)
        return inv_sqrt.unsqueeze(1) * adj * inv_sqrt.unsqueeze(0)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor, claim: torch.Tensor) -> torch.Tensor:
        logits_list = []
        batch_size = embeddings.size(0)

        for idx in range(batch_size):
            valid = mask[idx] > 0
            evidence = embeddings[idx][valid]
            claim_node = claim[idx].unsqueeze(0)

            if evidence.numel() == 0:
                nodes = claim_node
            else:
                nodes = torch.cat([claim_node, evidence], dim=0)

            adj = self._build_adj(nodes)
            hidden = torch.relu(self.input_proj(nodes))

            for layer in self.gcn_layers:
                hidden = torch.relu(adj @ layer(hidden))
                hidden = self.dropout(hidden)

            claim_repr = hidden[0]
            if hidden.size(0) > 1:
                evidence_repr = hidden[1:].mean(dim=0)
            else:
                evidence_repr = claim_repr

            graph_repr = torch.cat([claim_repr, evidence_repr], dim=-1)
            logits_list.append(self.classifier(graph_repr))

        return torch.stack(logits_list, dim=0)


class DeClarEClassifier(nn.Module):
    """DeClarE-style claim-aware attention over encoded evidence sequence."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.evidence_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        proj_dim = hidden_dim * 2
        self.claim_proj = nn.Linear(input_dim, proj_dim)

        self.attn_e = nn.Linear(proj_dim, proj_dim)
        self.attn_c = nn.Linear(proj_dim, proj_dim)
        self.attn_v = nn.Linear(proj_dim, 1)

        fused_dim = proj_dim * 4
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_labels),
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor, claim: torch.Tensor) -> torch.Tensor:
        evidence_seq = embeddings * mask.unsqueeze(-1)
        encoded, _ = self.evidence_encoder(evidence_seq)

        claim_repr = self.claim_proj(claim)
        attn_input = torch.tanh(self.attn_e(encoded) + self.attn_c(claim_repr).unsqueeze(1))
        scores = self.attn_v(attn_input).squeeze(-1)
        alpha = masked_softmax(scores, mask, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoded).squeeze(1)

        fused = torch.cat(
            [
                context,
                claim_repr,
                torch.abs(context - claim_repr),
                context * claim_repr,
            ],
            dim=-1,
        )
        fused = self.dropout(fused)
        return self.classifier(fused)


class EHIANClassifier(nn.Module):
    """Enhanced Hierarchical Interactive Attention Network for claim-evidence classification."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sentence_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        sent_dim = hidden_dim * 2
        # Claim-guided evidence attention
        self.ce_attn_e = nn.Linear(sent_dim, sent_dim)
        self.ce_attn_c = nn.Linear(sent_dim, sent_dim)
        self.ce_attn_v = nn.Linear(sent_dim, 1, bias=False)

        # Evidence-context attention
        self.ee_attn_e = nn.Linear(sent_dim, sent_dim)
        self.ee_attn_g = nn.Linear(sent_dim, sent_dim)
        self.ee_attn_v = nn.Linear(sent_dim, 1, bias=False)

        self.claim_proj = nn.Linear(input_dim, sent_dim)
        self.fuse_gate = nn.Linear(sent_dim * 3, sent_dim)

        fused_dim = sent_dim * 6
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, sent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sent_dim, num_labels),
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor, claim: torch.Tensor) -> torch.Tensor:
        sentence_inputs = embeddings * mask.unsqueeze(-1)
        encoded, _ = self.sentence_encoder(sentence_inputs)

        claim_repr = self.claim_proj(claim)

        # Branch-1: claim-guided attention over evidence sequence
        ce_scores = self.ce_attn_v(
            torch.tanh(self.ce_attn_e(encoded) + self.ce_attn_c(claim_repr).unsqueeze(1))
        ).squeeze(-1)
        ce_alpha = masked_softmax(ce_scores, mask, dim=1)
        evidence_ce = torch.bmm(ce_alpha.unsqueeze(1), encoded).squeeze(1)

        # Branch-2: evidence-context attention for intra-evidence consistency
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        evidence_global = (encoded * mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        ee_scores = self.ee_attn_v(
            torch.tanh(self.ee_attn_e(encoded) + self.ee_attn_g(evidence_global).unsqueeze(1))
        ).squeeze(-1)
        ee_alpha = masked_softmax(ee_scores, mask, dim=1)
        evidence_ee = torch.bmm(ee_alpha.unsqueeze(1), encoded).squeeze(1)

        gate = torch.sigmoid(self.fuse_gate(torch.cat([evidence_ce, evidence_ee, claim_repr], dim=-1)))
        evidence_repr = gate * evidence_ce + (1.0 - gate) * evidence_ee

        fused = torch.cat(
            [
                evidence_repr,
                claim_repr,
                torch.abs(evidence_repr - claim_repr),
                evidence_repr * claim_repr,
                evidence_ce,
                evidence_ee,
            ],
            dim=-1,
        )
        fused = self.dropout(fused)
        return self.classifier(fused)


class GETLayer(nn.Module):
    """Graph-enhanced Transformer layer for evidence-aware interactions."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor, node_mask: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        node_mask: (B, N), 1 for valid node
        attn_bias: (B, N, N), additive structural bias
        """
        batch_size, num_nodes, hidden_dim = x.shape

        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attn_bias.unsqueeze(1)

        key_mask = node_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(key_mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        query_mask = node_mask.unsqueeze(1).unsqueeze(-1)
        attn = attn * query_mask
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, hidden_dim)

        x = self.norm1(x + self.dropout(self.o_proj(context)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class GETClassifier(nn.Module):
    """
    Evidence-aware Fake News Detection with Graph Neural Networks (GET-style baseline).

    This implementation models claim/evidence nodes with graph-enhanced Transformer layers
    and injects evidence-aware structural bias in attention.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([GETLayer(hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        # Learnable coefficients for structural attention bias.
        self.sim_scale = nn.Parameter(torch.tensor(0.50))
        self.claim_edge_scale = nn.Parameter(torch.tensor(0.15))

        fused_dim = hidden_dim * 5
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor, claim: torch.Tensor) -> torch.Tensor:
        batch_size = embeddings.size(0)
        claim_node = claim.unsqueeze(1)
        node_inputs = torch.cat([claim_node, embeddings], dim=1)  # (B, 6, D)

        node_mask = torch.cat(
            [torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype), mask],
            dim=1,
        ).float()

        # Build evidence-aware structural bias from node cosine similarity + claim-evidence prior.
        node_norm = F.normalize(node_inputs, dim=-1)
        sim_bias = torch.matmul(node_norm, node_norm.transpose(1, 2))

        num_nodes = node_inputs.size(1)
        claim_prior = torch.zeros(num_nodes, num_nodes, device=node_inputs.device, dtype=node_inputs.dtype)
        if num_nodes > 1:
            claim_prior[0, 1:] = 1.0
            claim_prior[1:, 0] = 1.0

        attn_bias = self.sim_scale * sim_bias + self.claim_edge_scale * claim_prior.unsqueeze(0)
        valid_pair = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        attn_bias = attn_bias * valid_pair

        x = self.input_proj(node_inputs)
        x = x * node_mask.unsqueeze(-1)

        for layer in self.layers:
            x = layer(x, node_mask, attn_bias)
            x = x * node_mask.unsqueeze(-1)

        claim_repr = x[:, 0, :]
        evidence_nodes = x[:, 1:, :]
        ev_mask = mask.float()
        ev_denom = ev_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)

        evidence_mean = (evidence_nodes * ev_mask.unsqueeze(-1)).sum(dim=1) / ev_denom

        # Claim-guided evidence context.
        ce_scores = torch.sum(evidence_nodes * claim_repr.unsqueeze(1), dim=-1) / math.sqrt(claim_repr.size(-1))
        ce_alpha = masked_softmax(ce_scores, ev_mask, dim=1)
        evidence_ctx = torch.bmm(ce_alpha.unsqueeze(1), evidence_nodes).squeeze(1)

        fused = torch.cat(
            [
                claim_repr,
                evidence_mean,
                evidence_ctx,
                torch.abs(claim_repr - evidence_ctx),
                claim_repr * evidence_ctx,
            ],
            dim=-1,
        )
        fused = self.dropout(fused)
        return self.classifier(fused)


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=2, ensure_ascii=False)


def mean_std(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "values": []}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "values": [float(v) for v in array.tolist()],
    }


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_labels: int,
    target_names: List[str],
) -> Dict[str, Any]:
    model.eval()
    all_preds, all_golds = [], []
    with torch.no_grad():
        for batch in loader:
            emb = batch["embeddings"].to(device)
            msk = batch["mask"].to(device)
            clm = batch["claim"].to(device)
            logits = model(emb, msk, clm)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_golds.extend(batch["label"].tolist())

    report = classification_report(
        all_golds,
        all_preds,
        labels=list(range(num_labels)),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "classification_report": report,
        "macro_f1": float(f1_score(all_golds, all_preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(all_golds, all_preds, average="micro", zero_division=0)),
        "accuracy": float(accuracy_score(all_golds, all_preds)),
    }


def build_target_names(num_labels: int) -> List[str]:
    if num_labels == 2:
        return ["Real", "Fake"]
    return [f"Class_{idx}" for idx in range(num_labels)]


def build_model(model_name: str, input_dim: int, num_labels: int) -> nn.Module:
    if model_name == "textcnn":
        return TextCNNClassifier(input_dim=input_dim, num_labels=num_labels)
    if model_name == "induct_gcn":
        return InducTGCNClassifier(input_dim=input_dim, num_labels=num_labels)
    if model_name == "declare":
        return DeClarEClassifier(input_dim=input_dim, num_labels=num_labels)
    if model_name in {"ehian", "han"}:
        return EHIANClassifier(input_dim=input_dim, num_labels=num_labels)
    if model_name == "get":
        return GETClassifier(input_dim=input_dim, num_labels=num_labels)
    raise ValueError(f"unknown model_name: {model_name}")


def train_and_eval_variant(
    model_name: str,
    variant: str,
    cache_dir: str = "cache",
    num_runs: int = 5,
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 1e-4,
    device: torch.device = None,
    out_dir: str = "outputs/baseline_new",
):
    selection_strategy = "last_epoch"
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NpyFactCheckDataset(cache_dir, variant=variant)
    num_labels = dataset.num_labels
    target_names = build_target_names(num_labels)
    variant_name = "before_replace" if variant == "prev" else "after_replace"

    all_reports = []
    all_runs_macro_f1 = []
    all_runs_micro_f1 = []
    all_runs_acc = []
    run_summaries = []

    variant_output_dir = Path(out_dir) / model_name / variant_name
    variant_output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting {num_runs} runs on {device} "
        f"(model={model_name}, variant={variant}, num_labels={num_labels})..."
    )

    for run in range(num_runs):
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        run_output_dir = variant_output_dir / f"run_{run + 1:02d}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        epoch_logs = []

        model = build_model(model_name=model_name, input_dim=dataset.embedding_dim, num_labels=num_labels).to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        eval_result = None

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            for batch in train_loader:
                emb = batch["embeddings"].to(device)
                msk = batch["mask"].to(device)
                clm = batch["claim"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                logits = model(emb, msk, clm)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                num_batches += 1

            train_loss = epoch_loss / max(num_batches, 1)
            eval_result = evaluate_loader(
                model=model,
                loader=val_loader,
                device=device,
                num_labels=num_labels,
                target_names=target_names,
            )
            epoch_logs.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_accuracy": eval_result["accuracy"],
                    "val_macro_f1": eval_result["macro_f1"],
                    "val_micro_f1": eval_result["micro_f1"],
                }
            )

        if eval_result is None:
            eval_result = evaluate_loader(
                model=model,
                loader=val_loader,
                device=device,
                num_labels=num_labels,
                target_names=target_names,
            )

        report = eval_result["classification_report"]
        all_reports.append(report)

        macro_f1 = float(eval_result["macro_f1"])
        micro_f1 = float(eval_result["micro_f1"])
        accuracy = float(eval_result["accuracy"])
        all_runs_macro_f1.append(macro_f1)
        all_runs_micro_f1.append(micro_f1)
        all_runs_acc.append(accuracy)

        run_summary = {
            "run_index": run + 1,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "classification_report": report,
        }
        run_summaries.append(run_summary)

        write_json(
            run_output_dir / "epoch_log.json",
            {
                "model": model_name,
                "variant": variant,
                "variant_name": variant_name,
                "selection_strategy": selection_strategy,
                "run_index": run + 1,
                "epochs": epoch_logs,
            },
        )
        write_json(
            run_output_dir / "result.json",
            {
                "model": model_name,
                "variant": variant,
                "variant_name": variant_name,
                "run_index": run + 1,
                "final": run_summary,
            },
        )

        print(
            f"Run {run + 1}/{num_runs} completed. "
            f"acc={accuracy:.4f}, macro_f1={macro_f1:.4f}, micro_f1={micro_f1:.4f}"
        )

    def get_stat(key_path: str, metric: str) -> List[float]:
        values = []
        for item in all_reports:
            if key_path in item and metric in item[key_path]:
                values.append(item[key_path][metric])
        return values

    def format_cell(values: List[float]) -> str:
        if not values:
            return "N/A"
        return f"{np.mean(values):.4f} +/- {np.std(values):.4f}"

    print("\n" + "=" * 72)
    print(
        f"[BaselineNew] Final Report (model={model_name}, {variant_name}, "
        f"runs={num_runs})"
    )
    print("=" * 72)
    print(f"Macro-F1 Avg: {format_cell(all_runs_macro_f1)}")
    print(f"Micro-F1 Avg: {format_cell(all_runs_micro_f1)}")
    print("-" * 72)

    headers = ["Category", "Precision", "Recall", "F1-score"]
    print(f"{headers[0]:<15} {headers[1]:<20} {headers[2]:<20} {headers[3]:<20}")

    for label in target_names + ["macro avg", "weighted avg"]:
        p = get_stat(label, "precision")
        r = get_stat(label, "recall")
        f = get_stat(label, "f1-score")
        print(f"{label:<15} {format_cell(p):<20} {format_cell(r):<20} {format_cell(f):<20}")

    acc_list = [rep["accuracy"] for rep in all_reports if "accuracy" in rep]
    print(f"{'accuracy':<15} {' ':<20} {' ':<20} {format_cell(acc_list):<20}")
    print("=" * 72)

    summary_payload = {
        "model": model_name,
        "variant": variant,
        "variant_name": variant_name,
        "selection_strategy": selection_strategy,
        "config": {
            "cache_dir": cache_dir,
            "num_runs": num_runs,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
        },
        "aggregate": {
            "accuracy": mean_std(all_runs_acc),
            "macro_f1": mean_std(all_runs_macro_f1),
            "micro_f1": mean_std(all_runs_micro_f1),
        },
        "per_label": {},
        "runs": run_summaries,
    }

    for label in target_names + ["macro avg", "weighted avg"]:
        summary_payload["per_label"][label] = {
            "precision": mean_std(get_stat(label, "precision")),
            "recall": mean_std(get_stat(label, "recall")),
            "f1_score": mean_std(get_stat(label, "f1-score")),
        }

    write_json(variant_output_dir / "summary.json", summary_payload)
    print(f"[BaselineNew] JSON exported to {variant_output_dir}")


def resolve_variants_for_model(model_name: str, requested_variants: List[str]) -> List[str]:
    unique_variants = list(dict.fromkeys(requested_variants))
    if model_name in {"textcnn", "induct_gcn"}:
        if unique_variants != ["r"]:
            print(
                f"[BaselineNew] model={model_name} only runs optimized variant 'r'; "
                f"requested={unique_variants} ignored."
            )
        return ["r"]
    return unique_variants


def train_and_eval(
    models: List[str],
    variants: List[str],
    cache_dir: str = "cache",
    num_runs: int = 5,
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 1e-4,
    out_dir: str = "outputs/baseline",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_name in models:
        model_variants = resolve_variants_for_model(model_name, variants)
        for variant in model_variants:
            train_and_eval_variant(
                model_name=model_name,
                variant=variant,
                cache_dir=cache_dir,
                num_runs=num_runs,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                device=device,
                out_dir=out_dir,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline models: TextCNN / InducT-GCN / DeClarE / EHIAN / GET")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "textcnn", "induct_gcn", "declare", "ehian", "han", "get"],
        help="single model selector (compatible mode)",
    )
    parser.add_argument("--enable-textcnn", action="store_true", help="enable TextCNN")
    parser.add_argument("--enable-induct-gcn", action="store_true", help="enable InducT-GCN")
    parser.add_argument("--enable-declare", action="store_true", help="enable DeClarE")
    parser.add_argument("--enable-ehian", action="store_true", help="enable EHIAN")
    parser.add_argument("--enable-get", action="store_true", help="enable GET")
    parser.add_argument("--enable-han", action="store_true", help="deprecated alias of --enable-ehian")
    parser.add_argument(
        "--variant",
        type=str,
        default="both",
        choices=["both", "prev", "r"],
        help="evidence embedding variant",
    )
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=str, default="outputs/baseline_new")
    return parser.parse_args()


def resolve_models(args) -> List[str]:
    """Resolve model list from fine-grained switches, fallback to --model."""
    toggled_models = []
    if args.enable_textcnn:
        toggled_models.append("textcnn")
    if args.enable_induct_gcn:
        toggled_models.append("induct_gcn")
    if args.enable_declare:
        toggled_models.append("declare")
    if args.enable_ehian or args.enable_han:
        toggled_models.append("ehian")
    if args.enable_get:
        toggled_models.append("get")

    if toggled_models:
        # Keep insertion order while removing duplicates.
        return list(dict.fromkeys(toggled_models))

    if args.model == "all":
        return ["textcnn", "induct_gcn", "declare", "ehian", "get"]
    if args.model == "han":
        return ["ehian"]
    return [args.model]


def main():
    args = parse_args()
    models = resolve_models(args)
    variants = ["prev", "r"] if args.variant == "both" else [args.variant]

    train_and_eval(
        models=models,
        variants=variants,
        cache_dir=args.cache_dir,
        num_runs=args.num_runs,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()