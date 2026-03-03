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

    Feature order:
      0: relevance (cosine between claim and evidence)
      1: entail_ce
      2: contra_ce
      3: ee_support (mean entail_ee over other evidences)
      4: ee_conflict (mean contra_ee over other evidences)
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
    num_evd = evd_mask.shape[1]
    pair_mask = evd_mask[:, :, None] * evd_mask[:, None, :]
    eye = np.eye(num_evd, dtype=np.float32)[None, :, :]
    pair_mask = pair_mask * (1.0 - eye)

    entail_ee = nli_ee[:, :, :, entail_idx] * pair_mask
    contra_ee = nli_ee[:, :, :, contra_idx] * pair_mask

    denom = pair_mask.sum(axis=2)
    ee_support = np.zeros_like(denom, dtype=np.float32)
    ee_conflict = np.zeros_like(denom, dtype=np.float32)
    np.divide(entail_ee.sum(axis=2), denom, out=ee_support, where=denom > 0)
    np.divide(contra_ee.sum(axis=2), denom, out=ee_conflict, where=denom > 0)

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
