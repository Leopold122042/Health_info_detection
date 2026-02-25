"""
models_gnn.py - 异构图神经网络主模型
功能：整合一致性模块、门控模块，实现 CA-HGER 架构
创新点：异构图编码、检索门控、一致性感知
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import get_config
from modules_consistency import RetrievalGatingModule, ConsistencyAttentionModule, ConsistencyLossModule


class HeteroGNNLayer(nn.Module):
    """
    异构图神经网络层
    功能：根据 edge_type 使用不同的权重矩阵进行消息传递
    对应文献 1：Heterogeneous GNN for cross-document reasoning
    """
    
    def __init__(self, emb_dim, num_edge_types, config=None):
        super().__init__()
        self.config = config if config else get_config()
        self.emb_dim = emb_dim
        self.num_edge_types = num_edge_types
        
        # 为每种边类型学习独立的权重矩阵
        # 边类型定义 (Phase 1): 0=C-E, 1=E-E, 2=Self-loop
        self.relation_weights = nn.ModuleList([
            nn.Linear(emb_dim, emb_dim, bias=False) for _ in range(num_edge_types)
        ])
        
        # 注意力模块 (用于 E-E 边的一致性建模)
        self.consistency_attn = ConsistencyAttentionModule(emb_dim, num_heads=4, config=config)
        
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(emb_dim)
        
    def forward(self, node_features, edge_index, edge_type, edge_weight=None):
        """
        Args:
            node_features: (total_nodes, emb_dim)
            edge_index: (2, num_edges)
            edge_type: (num_edges,) 长整型，0, 1, 2
            edge_weight: (num_edges,) 可选
            
        Returns:
            output: (total_nodes, emb_dim)
        """
        total_nodes = node_features.size(0)
        num_edges = edge_index.size(1)
        
        # 初始化输出
        output = torch.zeros_like(node_features)
        
        # 按边类型聚合消息
        for r in range(self.num_edge_types):
            # 获取当前类型的边掩码
            mask = (edge_type == r)
            if not mask.any():
                continue
            
            # 筛选边
            r_edge_index = edge_index[:, mask]  # (2, num_r_edges)
            r_edge_weight = edge_weight[mask] if edge_weight is not None else None
            
            # 获取源节点特征
            src_idx = r_edge_index[0]
            dst_idx = r_edge_index[1]
            src_features = node_features[src_idx]
            
            # 应用关系特定变换
            transformed = self.relation_weights[r](src_features)  # (num_r_edges, emb_dim)
            
            # 如果是 E-E 边 (type=1)，应用一致性注意力
            if r == 1: 
                # 这里简化处理：将变换后的特征作为消息，结合一致性权重
                # 实际应调用 consistency_attn，但为了性能，这里将 edge_weight 作为注意力权重
                if r_edge_weight is not None:
                    # 归一化权重
                    attn_weights = (r_edge_weight + 1) / 2  # -1~1 -> 0~1
                    messages = transformed * attn_weights.unsqueeze(-1)
                else:
                    messages = transformed
            else:
                messages = transformed
            
            # 聚合到目标节点 (sum)
            output.index_add_(0, dst_idx, messages)
        
        # 残差连接 + 归一化 + 激活
        output = output + node_features  # Residual
        output = self.bn(output)
        output = F.relu(output)
        output = self.dropout(output)
        
        return output


class CA_HGER_Model(nn.Module):
    """
    Consistency-Aware Heterogeneous Graph Evidence Reasoning Network
    主模型类
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config else get_config()
        
        # 参数
        self.emb_dim = self.config.model.emb_dim
        self.hidden_dim = self.config.model.hidden_dim
        self.gnn_layers = self.config.model.gnn_layers
        self.num_labels = 2  # 修正：Phase 1 统计显示为 2 分类
        self.num_edge_types = 3  # 0=C-E, 1=E-E, 2=Self-loop
        
        # 1. 检索证据门控 (输入层)
        self.retrieval_gate = RetrievalGatingModule(self.emb_dim, config=self.config)
        
        # 2. 异构图编码器 (GNN 层)
        self.gnn_layers = nn.ModuleList([
            HeteroGNNLayer(self.emb_dim, self.num_edge_types, config=self.config)
            for _ in range(self.gnn_layers)
        ])
        
        # 3. 一致性损失模块 (辅助)
        self.consistency_loss_module = ConsistencyLossModule(config=self.config)
        
        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.emb_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, self.num_labels)
        )
        
    def forward(self, batch_data):
        """
        Args:
            batch_data: Dict (来自 BatchedGraphCollator)
                - node_features: (total_nodes, emb_dim)
                - edge_index: (2, num_edges)
                - edge_type: (num_edges,)
                - edge_weight: (num_edges,)
                - retrieval_flags: (total_nodes,)
                - claim_node_indices: (batch_size,)
                - labels: (batch_size,) [训练时]
                
        Returns:
            logits: (batch_size, num_labels)
            consistency_loss: scalar (可选)
            attention_weights: (num_edges,) (可选，用于可解释性)
        """
        # 1. 提取数据
        node_features = batch_data['node_features']
        edge_index = batch_data['edge_index']
        edge_type = batch_data['edge_type']
        edge_weight = batch_data.get('edge_weight', None)
        retrieval_flags = batch_data.get('retrieval_flags', None)
        claim_indices = batch_data['claim_node_indices']
        
        # 2. 检索证据门控 (创新点 1)
        if retrieval_flags is not None:
            node_features, gate_values = self.retrieval_gate(node_features, retrieval_flags)
        else:
            gate_values = None
        
        # 3. 图神经网络推理 (创新点 2)
        hidden = node_features
        for i, gnn_layer in enumerate(self.gnn_layers):
            hidden = gnn_layer(hidden, edge_index, edge_type, edge_weight)
        
        # 4. 读出层：提取声明节点嵌入
        claim_embeddings = hidden[claim_indices]  # (batch_size, emb_dim)
        
        # 5. 分类
        logits = self.classifier(claim_embeddings)  # (batch_size, num_labels)
        
        # 6. 一致性损失计算 (创新点 3，可选)
        # 这里简化：仅计算所有证据节点的方差作为一致性代理
        # 实际应区分每个样本的证据节点
        consistency_loss = torch.tensor(0.0, device=logits.device)
        if self.config.model.use_consistency_loss and self.training:
            # 需要更复杂的逻辑来按样本分割证据节点，此处暂略，留待 Phase 3 完善
            pass
        
        return logits, consistency_loss, gate_values
    
    def predict(self, batch_data):
        """推理模式"""
        self.eval()
        with torch.no_grad():
            logits, _, _ = self.forward(batch_data)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs