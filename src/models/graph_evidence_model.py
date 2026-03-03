from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GraphBatch:
    """
    与 GraphSample 对应的轻量包装，用于模型前向。

    当前实现中，我们仍按“每次一个小图”来处理；
    如需真正 batch 多图，可在后续扩展为拼接 + batch_index。
    """

    node_feats: torch.Tensor  # (num_nodes, node_dim)
    edge_index: torch.Tensor  # (2, num_edges)
    edge_feats: torch.Tensor  # (num_edges, edge_dim)


class EdgeAwareGraphLayer(nn.Module):
    """
    简单的基于边特征的图消息传递层：
      - 使用 edge_feats 生成注意力/权重；
      - 对邻居节点聚合后，再与自身节点特征融合。
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.msg_mlp = nn.Linear(node_dim + edge_dim, hidden_dim)
        self.out_mlp = nn.Linear(node_dim + hidden_dim, node_dim)

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor, edge_feats: torch.Tensor):
        if edge_index.numel() == 0:
            return node_feats

        src, dst = edge_index  # (E,), (E,)
        src_feats = node_feats[src]  # (E, node_dim)

        # 基于边特征的消息构造
        msg_input = torch.cat([src_feats, edge_feats], dim=-1)
        msgs = F.relu(self.msg_mlp(msg_input))  # (E, hidden_dim)

        num_nodes = node_feats.size(0)
        agg = torch.zeros(num_nodes, msgs.size(-1), device=node_feats.device)
        agg.index_add_(0, dst, msgs)

        out = torch.cat([node_feats, agg], dim=-1)
        out = F.relu(self.out_mlp(out))
        return out


class GraphEvidenceReasoner(nn.Module):
    """
    图证据推理网络：
      - 节点 0 为 Claim，1..M 为 Evidence
      - 依赖 GraphFactCheckDataset 构造的 node_feats / edge_feats
      - 通过若干层 EdgeAwareGraphLayer 建模 CE / EE-Int / TF-IDF 等结构化信息
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [EdgeAwareGraphLayer(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, batch: GraphBatch):
        x = batch.node_feats
        edge_index = batch.edge_index
        edge_feats = batch.edge_feats

        for layer in self.layers:
            x = layer(x, edge_index, edge_feats)
            x = self.dropout(x)

        # 使用 Claim 节点（索引 0）的表征进行分类
        claim_repr = x[0]
        logits = self.classifier(claim_repr)
        return logits


def build_graph_batch_from_sample(sample) -> GraphBatch:
    """
    将 GraphSample 转为模型可直接接受的 GraphBatch。
    """
    return GraphBatch(
        node_feats=sample.node_feats,
        edge_index=sample.edge_index,
        edge_feats=sample.edge_feats,
    )

