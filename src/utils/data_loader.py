import numpy as np
import torch
from pathlib import Path
from preprocess.gen_evidence_feats import build_evidence_features_from_arrays
from torch.utils.data import Dataset, DataLoader

class HealthGraphDataset(Dataset):
    def __init__(self, cache_dir="cache"):
        # 加载所有预处理好的特征
        self.claim_embs = np.load(f"{cache_dir}/claims_embeddings.npy")       # (N, 768)
        self.evid_embs = np.load(f"{cache_dir}/evidences_embeddings_r.npy")         # (N, 5, 768)
        self.nli_ce = np.load(f"{cache_dir}/nli_logits_ce.npy")        # (N, 5, 3)
        self.nli_ee = np.load(f"{cache_dir}/nli_logits_ee.npy")        # (N, 5, 5, 3)
        self.tfidf_w = np.load(f"{cache_dir}/tfidf_weights.npy")       # (N, 5, 1)
        self.masks = np.load(f"{cache_dir}/evd_mask_r.npy")                 # (N, 5)
        self.labels = np.load(f"{cache_dir}/labels.npy")               # (N,)

        evd_feats_path = Path(cache_dir) / "evidence_feats.npy"
        if evd_feats_path.exists():
            self.evd_feats = np.load(evd_feats_path)  # (N, 5, 5)
        else:
            # Build on the fly if not precomputed
            self.evd_feats = build_evidence_features_from_arrays(
                self.claim_embs,
                self.evid_embs,
                self.nli_ce,
                self.nli_ee,
                self.masks,
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "claim": torch.tensor(self.claim_embs[idx], dtype=torch.float32),
            "evidences": torch.tensor(self.evid_embs[idx], dtype=torch.float32),
            "nli_ce": torch.tensor(self.nli_ce[idx], dtype=torch.float32),
            "nli_ee": torch.tensor(self.nli_ee[idx], dtype=torch.float32),
            "tfidf": torch.tensor(self.tfidf_w[idx], dtype=torch.float32),
            "evd_feats": torch.tensor(self.evd_feats[idx], dtype=torch.float32),
            "mask": torch.tensor(self.masks[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def get_dataloader(batch_size=32, shuffle=True):
    dataset = HealthGraphDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)