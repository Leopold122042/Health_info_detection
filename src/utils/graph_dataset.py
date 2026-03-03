import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class GraphSample:
    """
    单个 Claim-Evidence 图样本的数据结构。

    - node_feats: (num_nodes, node_dim) 节点特征，0 为 claim，其余为 evidences
    - edge_index: (2, num_edges) long，源和目标节点索引
    - edge_feats: (num_edges, edge_dim) 边特征（可为空张量）
    - label: 标量类别 id
    - num_evidences: 实际有效证据数量（不含 claim）
    """
    node_feats: torch.Tensor
    edge_index: torch.Tensor
    edge_feats: torch.Tensor
    label: torch.Tensor
    num_evidences: int


class GraphFactCheckDataset(Dataset):
    """
    将 cache 目录下的 numpy 特征组织为 Claim-Evidence 图数据集。

    依赖的文件（全部位于 cache_dir）：
      - labels.npy                  -> (N,)
      - evidences_embeddings_r.npy  -> (N, 5, d_emb)
      - evd_mask_r.npy              -> (N, 5)
      - claims_embeddings.npy       -> (N, d_emb)           （可选，若不存在则用证据平均代替）
      - evidence_feats.npy          -> (N, 5, d_feat)       （CE/EE-Int 特征）
      - tfidf_weights.npy           -> (N, 5, 1)            （可选）
    """

    def __init__(
        self,
        cache_dir: str = "cache",
        use_tfidf: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.use_tfidf = use_tfidf
        self.device = device or torch.device("cpu")

        cache_path = os.path.abspath(cache_dir)

        self.labels = np.load(os.path.join(cache_path, "labels.npy"))
        self.evid_embs = np.load(os.path.join(cache_path, "evidences_embeddings_r.npy"))
        self.evd_mask = np.load(os.path.join(cache_path, "evd_mask_r.npy"))
        self.evd_feats = np.load(os.path.join(cache_path, "evidence_feats.npy"))

        claims_path = os.path.join(cache_path, "claims_embeddings.npy")
        if os.path.exists(claims_path):
            self.claim_embs = np.load(claims_path)
        else:
            # 若没有单独的 claim 向量，则以有效证据平均向量近似
            mask = self.evd_mask[:, :, None]
            summed = (self.evid_embs * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), a_min=1e-6, a_max=None)
            self.claim_embs = summed / counts

        if self.use_tfidf:
            tfidf_path = os.path.join(cache_path, "tfidf_weights.npy")
            if os.path.exists(tfidf_path):
                self.tfidf = np.load(tfidf_path)
            else:
                # 若不存在 TF-IDF 文件，则退化为全 1 权重
                N, K, _ = self.evd_feats.shape
                self.tfidf = np.ones((N, K, 1), dtype=np.float32)
        else:
            self.tfidf = None

        assert (
            self.labels.shape[0]
            == self.evid_embs.shape[0]
            == self.evd_mask.shape[0]
            == self.evd_feats.shape[0]
            == self.claim_embs.shape[0]
        ), "cached arrays have inconsistent first dimension (N)"

        self.num_samples = self.labels.shape[0]
        self.num_evidences = self.evid_embs.shape[1]

    def __len__(self) -> int:
        return self.num_samples

    def _build_edges(
        self, valid_evd_idx: List[int], evd_feats: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构造当前样本的边结构和边特征。

        - Claim 记为节点 0
        - 有效 evidences 依次编号为 1..M
        - Claim-Evidence 边：0 <-> j
        - Evidence-Evidence 边：i <-> j（无向边按两条有向边表示）
        """
        edges_src = []
        edges_dst = []
        edge_features = []

        # 边特征统一为 5 维，避免 np.stack 时 shape 不一致：CE 用 (rel, entail_ce, contra_ce, 0, 0)，EE 用 (0,0,0, ee_support, ee_conflict)
        EDGE_DIM = 5

        # Claim-Evidence 边：前三维 (relevance, entail_ce, contra_ce)，后两维填 0
        for local_idx, evd_slot in enumerate(valid_evd_idx):
            node_id = local_idx + 1  # evidence 节点从 1 开始
            feat = evd_feats[evd_slot]  # (d_feat,) 至少 5
            ce_feat = np.zeros(EDGE_DIM, dtype=np.float32)
            ce_feat[:3] = feat[:3]

            # claim -> evidence
            edges_src.append(0)
            edges_dst.append(node_id)
            edge_features.append(ce_feat)

            # evidence -> claim
            edges_src.append(node_id)
            edges_dst.append(0)
            edge_features.append(ce_feat)

        # Evidence-Evidence 边：前三维填 0，后两维 (ee_support, ee_conflict)
        M = len(valid_evd_idx)
        for i in range(M):
            for j in range(i + 1, M):
                ni = i + 1
                nj = j + 1
                fi = evd_feats[valid_evd_idx[i]][3:5]
                fj = evd_feats[valid_evd_idx[j]][3:5]
                ee_feat = np.zeros(EDGE_DIM, dtype=np.float32)
                ee_feat[3:5] = (fi + fj) / 2.0

                edges_src.append(ni)
                edges_dst.append(nj)
                edge_features.append(ee_feat)
                edges_src.append(nj)
                edges_dst.append(ni)
                edge_features.append(ee_feat)

        if not edge_features:
            edge_feats = torch.zeros((0, EDGE_DIM), dtype=torch.float32, device=self.device)
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        else:
            edge_feats_np = np.stack(edge_features, axis=0)
            edge_feats = torch.from_numpy(edge_feats_np).float().to(self.device)
            edge_index = torch.tensor(
                [edges_src, edges_dst], dtype=torch.long, device=self.device
            )

        return edge_index, edge_feats

    def __getitem__(self, idx: int) -> GraphSample:
        label = int(self.labels[idx])
        claim_vec = self.claim_embs[idx]  # (d_emb,)
        evd_vecs = self.evid_embs[idx]  # (5, d_emb)
        evd_mask = self.evd_mask[idx]  # (5,)
        evd_feats = self.evd_feats[idx]  # (5, d_feat)

        # 有效证据槽位索引
        valid_evd_idx = [i for i in range(self.num_evidences) if evd_mask[i] == 1]

        # 节点特征统一维度：claim 与 evidence 均为 (d_emb + d_feat + [1])，claim 的 feat/tfidf 部分填 0
        d_emb = claim_vec.shape[0]
        d_feat = evd_feats.shape[-1]
        use_tf = self.tfidf is not None
        node_dim = d_emb + d_feat + (1 if use_tf else 0)

        node_features = []
        claim_ext = np.zeros(node_dim, dtype=np.float32)
        claim_ext[:d_emb] = claim_vec
        node_features.append(claim_ext)

        for i in valid_evd_idx:
            ev_vec = evd_vecs[i]
            feat_vec = evd_feats[i]
            if use_tf:
                tfidf_weight = self.tfidf[idx, i, 0]
                node_features.append(
                    np.concatenate([ev_vec, feat_vec, np.array([tfidf_weight], dtype=np.float32)])
                )
            else:
                node_features.append(np.concatenate([ev_vec, feat_vec]))

        node_feats_np = np.stack(node_features, axis=0)
        node_feats = torch.from_numpy(node_feats_np).float().to(self.device)

        edge_index, edge_feats = self._build_edges(valid_evd_idx, evd_feats)

        return GraphSample(
            node_feats=node_feats,
            edge_index=edge_index,
            edge_feats=edge_feats,
            label=torch.tensor(label, dtype=torch.long, device=self.device),
            num_evidences=len(valid_evd_idx),
        )


def graph_collate_fn(batch: List[GraphSample]):
    """
    简单的批处理函数：
    - 目前将每个样本视为独立小图，不做图间拼接，主要用于单样本前向或自定义循环；
    - 如需真正的 batched graph，可在后续引入 PyG/DGL 等库重新实现 collate。
    """
    return batch
