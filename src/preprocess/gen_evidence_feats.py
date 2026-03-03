import numpy as np
from pathlib import Path


def build_evidence_features_from_arrays(
    claim_embs: np.ndarray,
    evid_embs: np.ndarray,
    nli_ce: np.ndarray,
    nli_ee: np.ndarray,
    evd_mask: np.ndarray,
    entail_idx: int = 2,
    contra_idx: int = 0,
) -> np.ndarray:
    """Build 5-dim evidence features from existing arrays.

    Calibrated feature order:
      0: relevance      — cosine(claim, evidence)
      1: entail_ce      — P(ENTAILMENT | claim, evidence)
      2: contra_ce      — P(CONTRADICTION | claim, evidence)
      3: ee_support     — mean positive net-support from other evidences
                           net-support = max(0, sym_entail - sym_contra)
      4: ee_conflict    — mean positive net-conflict from other evidences
                           net-conflict = max(0, sym_contra - sym_entail)

    其中 sym_entail / sym_contra 通过对 (e_j, e_k) 与 (e_k, e_j) 的 NLI
    概率进行对称化平均，兼顾方向信息，避免单向噪声主导。
    """
    eps = 1e-12
    # relevance: cosine(claim, evidence)
    c = claim_embs[:, None, :]
    e = evid_embs
    c_norm = np.linalg.norm(c, axis=-1)
    e_norm = np.linalg.norm(e, axis=-1)
    relevance = (c * e).sum(axis=-1) / (c_norm * e_norm + eps)

    entail_ce = nli_ce[:, :, entail_idx]
    contra_ce = nli_ce[:, :, contra_idx]

    # evidence-evidence support/conflict (exclude self-pairs and missing evidences)
    # 使用对称化后的 NLI 概率，并基于“蕴含-矛盾”的净值构造更稳健的支持/冲突特征
    num_evd = evd_mask.shape[1]
    pair_mask = evd_mask[:, :, None] * evd_mask[:, None, :]
    eye = np.eye(num_evd, dtype=np.float32)[None, :, :]
    pair_mask = pair_mask * (1.0 - eye)

    # 原始有向 NLI 概率: p(e_j -> e_k)
    raw_entail = nli_ee[:, :, :, entail_idx]
    raw_contra = nli_ee[:, :, :, contra_idx]

    # 对称化: 同时考虑 p(e_j -> e_k) 与 p(e_k -> e_j)
    raw_entail_sym = 0.5 * (raw_entail + np.transpose(raw_entail, (0, 2, 1)))
    raw_contra_sym = 0.5 * (raw_contra + np.transpose(raw_contra, (0, 2, 1)))

    raw_entail_sym = raw_entail_sym * pair_mask
    raw_contra_sym = raw_contra_sym * pair_mask

    # 基于“蕴含-矛盾”的净值构建支持/冲突，正负部分分别聚合
    net_ee = raw_entail_sym - raw_contra_sym
    pos_support = np.maximum(net_ee, 0.0)
    neg_conflict = np.maximum(-net_ee, 0.0)

    denom = pair_mask.sum(axis=2)
    ee_support = np.zeros_like(denom, dtype=np.float32)
    ee_conflict = np.zeros_like(denom, dtype=np.float32)
    np.divide(pos_support.sum(axis=2), denom, out=ee_support, where=denom > 0)
    np.divide(neg_conflict.sum(axis=2), denom, out=ee_conflict, where=denom > 0)

    feats = np.stack([relevance, entail_ce, contra_ce, ee_support, ee_conflict], axis=-1)

    # zero out missing evidences
    feats = feats * evd_mask[:, :, None]
    return feats.astype(np.float32)


def generate_evidence_features(
    cache_dir: str = "cache",
    output_name: str = "evidence_feats.npy",
    entail_idx: int = 2,
    contra_idx: int = 0,
) -> None:
    """Generate evidence feature file from cached npy inputs."""
    cache = Path(cache_dir)
    claim_embs = np.load(cache / "claims_embeddings.npy")
    evid_embs = np.load(cache / "evidences_embeddings_r.npy")
    nli_ce = np.load(cache / "nli_logits_ce.npy")
    nli_ee = np.load(cache / "nli_logits_ee.npy")
    evd_mask = np.load(cache / "evd_mask_r.npy")

    feats = build_evidence_features_from_arrays(
        claim_embs,
        evid_embs,
        nli_ce,
        nli_ee,
        evd_mask,
        entail_idx=entail_idx,
        contra_idx=contra_idx,
    )

    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / output_name, feats)
    print(f"Saved evidence features to {cache / output_name}, shape={feats.shape}")


if __name__ == "__main__":
    generate_evidence_features()
