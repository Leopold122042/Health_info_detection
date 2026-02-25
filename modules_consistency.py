"""
modules_consistency.py - 一致性建模与检索门控模块
功能：实现证据间一致性计算、检索证据门控机制
创新点：检索证据置信度门控、证据一致性先验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import get_config


class RetrievalGatingModule(nn.Module):
    """
    检索证据门控模块
    功能：根据 retrieval_flags 动态调整证据节点的初始特征权重
    创新点：让模型自动学习是否信任检索到的证据
    """
    
    def __init__(self, emb_dim, config=None):
        super().__init__()
        self.config = config if config else get_config()
        self.emb_dim = emb_dim
        
        # 门控参数：用于检索证据 (flag=1)
        # 原始证据 (flag=0) 保持原样 (gate=1)
        # 声明节点 (flag=-1) 保持原样 (gate=1)
        self.retrieval_gate_weight = nn.Parameter(torch.ones(1))  # 可学习缩放因子
        self.retrieval_gate_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, node_features, retrieval_flags):
        """
        Args:
            node_features: (total_nodes, emb_dim) 所有节点特征
            retrieval_flags: (total_nodes,) 检索标记 (-1=claim, 0=original, 1=retrieved)
            
        Returns:
            gated_features: (total_nodes, emb_dim) 门控后的特征
        """
        # 创建门控掩码
        # 检索证据 (flag==1) 应用门控，其他保持 1
        is_retrieved = (retrieval_flags == 1).float().unsqueeze(-1)  # (total_nodes, 1)
        
        # 计算门控值：对于检索证据，学习一个缩放因子 [0, 1]
        # 使用 sigmoid 确保 gate 值在 0-1 之间，表示信任度
        gate_value = torch.sigmoid(self.retrieval_gate_weight * is_retrieved + self.retrieval_gate_bias)
        
        # 对于非检索证据，gate_value 应为 1 (因为 is_retrieved=0, sigmoid(bias) 可能不为 1)
        # 修正：直接构造 gate 向量
        # 原始逻辑：retrieved -> learnable gate, others -> 1.0
        gate = torch.ones_like(is_retrieved)
        # 仅对检索证据应用可学习门控 (这里简化为全局可学习参数，后续可升级为 per-node)
        # 更精细的实现：为检索证据学习一个独立的缩放系数
        retrieved_scale = torch.sigmoid(self.retrieval_gate_weight)
        gate = torch.where(is_retrieved > 0.5, retrieved_scale, torch.ones_like(is_retrieved))
        
        # 应用门控
        gated_features = node_features * gate
        
        return gated_features, gate


class ConsistencyAttentionModule(nn.Module):
    """
    一致性注意力模块
    功能：在 GNN 消息传递中，利用证据间相似度作为一致性先验
    创新点：结合文献 1 的跨文档推理，降低矛盾证据的注意力权重
    """
    
    def __init__(self, emb_dim, num_heads=4, config=None):
        super().__init__()
        self.config = config if config else get_config()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        
        # 注意力参数
        self.w_q = nn.Linear(emb_dim, emb_dim)
        self.w_k = nn.Linear(emb_dim, emb_dim)
        self.w_v = nn.Linear(emb_dim, emb_dim)
        
        # 一致性偏置参数 (用于融合 edge_weight 先验)
        self.consistency_bias_weight = nn.Parameter(torch.ones(1))
        
        self.dropout = nn.Dropout(0.5)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        
    def forward(self, node_features, edge_index, edge_weight=None, edge_mask=None):
        """
        Args:
            node_features: (total_nodes, emb_dim)
            edge_index: (2, num_edges)
            edge_weight: (num_edges,) 可选，Phase 1 计算的相似度先验
            edge_mask: (num_edges,) 可选，用于屏蔽某些边
            
        Returns:
            updated_features: (total_nodes, emb_dim)
            attention_weights: (num_edges,) 可用于可解释性分析
        """
        total_nodes = node_features.size(0)
        num_edges = edge_index.size(1)
        
        # 1. 计算 Q, K, V
        Q = self.w_q(node_features)  # (N, D)
        K = self.w_k(node_features)  # (N, D)
        V = self.w_v(node_features)  # (N, D)
        
        # 2. 获取边两端的节点特征
        src_idx = edge_index[0]  # (E,)
        dst_idx = edge_index[1]  # (E,)
        
        q_src = Q[src_idx]  # (E, D)
        k_dst = K[dst_idx]  # (E, D)
        v_dst = V[dst_idx]  # (E, D)
        
        # 3. 计算注意力分数 (点积)
        # 简化版单头注意力计算，多头可 reshape 后类似操作
        attention_scores = (q_src * k_dst).sum(dim=-1) / (self.head_dim ** 0.5)  # (E,)
        
        # 4. 融合一致性先验 (edge_weight)
        # 如果 edge_weight 存在 (证据间相似度)，将其作为 bias 加入
        # 相似度高 -> 一致性高 -> 注意力应更高
        if edge_weight is not None:
            # 归一化 edge_weight 到 0-1 范围 (Phase 1 余弦相似度已在 -1 到 1)
            # 映射到 0-1: (sim + 1) / 2
            normalized_weight = (edge_weight + 1) / 2
            # 应用一致性偏置
            consistency_bias = self.consistency_bias_weight * normalized_weight
            attention_scores = attention_scores + consistency_bias
        
        # 5. 应用掩码 (如果有)
        if edge_mask is not None:
            attention_scores = attention_scores.masked_fill(~edge_mask, -1e9)
            
        # 6. Softmax (需要在目标节点维度归一化，这里简化为边级别 softmax)
        # 由于是稀疏边列表，标准 GAT 需要对每个 dst 节点的 incoming edges 做 softmax
        # 这里使用简化版：直接 softmax 所有边，或使用 scatter_softmax
        # 为了稳定性，使用 torch_geometric 风格的 scatter_softmax (手动实现)
        attention_weights = self._scatter_softmax(attention_scores, dst_idx, num_nodes=total_nodes)
        
        # 7. 聚合消息
        messages = v_dst * attention_weights.unsqueeze(-1)  # (E, D)
        
        # 8. 聚合到节点 (sum aggregation)
        updated_features = torch.zeros_like(node_features)
        updated_features.index_add_(0, dst_idx, messages)
        
        # 9. 输出投影
        output = self.out_proj(updated_features)
        
        return output, attention_weights
    
    def _scatter_softmax(self, src, index, num_nodes):
        """
        对每个目标节点的入边进行 softmax
        """
        # 计算最大值用于数值稳定
        max_val = torch.zeros(num_nodes, device=src.device).index_reduce_(0, index, src, reduce='amax')
        # 减去最大值
        src_exp = (src - max_val[index]).exp()
        # 计算分母 (sum)
        sum_val = torch.zeros(num_nodes, device=src.device).index_reduce_(0, index, src_exp, reduce='sum')
        # 归一化
        return src_exp / (sum_val[index] + 1e-9)


class ConsistencyLossModule(nn.Module):
    """
    一致性损失计算模块 (辅助任务)
    功能：计算证据间的一致性损失，用于 Phase 3 的多任务学习
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config else get_config()
        
    def forward(self, evidence_embeddings, evidence_labels=None):
        """
        计算证据嵌入之间的方差作为一致性损失
        假说：真新闻的证据嵌入应更聚集，假新闻更分散
        """
        if evidence_embeddings.size(0) < 2:
            return torch.tensor(0.0, device=evidence_embeddings.device)
        
        # 计算证据嵌入的方差 (作为不一致性的代理)
        variance = torch.var(evidence_embeddings, dim=0).mean()
        
        return variance