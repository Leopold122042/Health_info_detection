"""
graph_builder.py - 异构图构建脚本
功能：构建声明 - 证据异构图，包括证据间一致性边
创新点：证据 - 证据边建模、检索证据标记
"""

import torch
import numpy as np
import os
from typing import Dict, List, Tuple
from config import get_config, Config


class GraphBuilder:
    """
    异构图构建器
    构建包含以下节点的图：
    - 1 个声明节点
    - N 个证据节点 (最多 5 个)
    
    构建以下边：
    - 声明->证据 (Claim-Evidence)
    - 证据->证据 (Evidence-Evidence) [核心创新]
    - 自环 (Self-loop)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.emb_dim = config.model.emb_dim
        self.max_evidences = config.model.max_evidences
        
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        计算证据间的相似度矩阵
        
        Args:
            embeddings: (num_evidences, emb_dim) 证据嵌入
            
        Returns:
            similarity_matrix: (num_evidences, num_evidences) 相似度矩阵
        """
        # 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-9, None)  # 避免除零
        normalized = embeddings / norms
        
        # 余弦相似度
        similarity = np.dot(normalized, normalized.T)
        
        # 将对角线设为 0 (排除自相似)
        np.fill_diagonal(similarity, 0)
        
        return similarity
    
    def build_evidence_evidence_edges(
        self, 
        evidence_emb: np.ndarray, 
        evidence_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建证据 - 证据边
        
        Args:
            evidence_emb: (max_evidences, emb_dim) 证据嵌入
            evidence_mask: (max_evidences,) 证据掩码 (1=有效，0=padding)
            
        Returns:
            edge_index: (2, num_edges) 边索引
            edge_weight: (num_edges,) 边权重 (相似度)
        """
        # 获取有效证据索引
        valid_indices = np.where(evidence_mask > 0.5)[0]
        num_valid = len(valid_indices)
        
        if num_valid < 2:
            # 少于 2 个证据，无法构建 E-E 边
            return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
        
        # 提取有效证据嵌入
        valid_emb = evidence_emb[valid_indices]
        
        # 计算相似度矩阵
        sim_matrix = self.compute_similarity_matrix(valid_emb)
        
        # 阈值过滤
        threshold = self.config.graph.ee_edge_threshold
        edge_mask = sim_matrix > threshold
        
        # 获取边索引 (在上三角矩阵中，避免重复)
        edge_rows, edge_cols = np.where(np.triu(edge_mask, k=1))
        
        if len(edge_rows) == 0:
            return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
        
        # 映射回原始索引
        src_indices = valid_indices[edge_rows]
        dst_indices = valid_indices[edge_cols]
        
        # 构建双向边
        edge_index = np.concatenate([
            np.stack([src_indices, dst_indices], axis=0),
            np.stack([dst_indices, src_indices], axis=0)
        ], axis=1)
        
        # 边权重
        edge_weights = sim_matrix[edge_rows, edge_cols]
        edge_weights = np.concatenate([edge_weights, edge_weights], axis=0)
        
        return edge_index.astype(np.int64), edge_weights.astype(np.float32)
    
    def build_claim_evidence_edges(
        self, 
        num_evidences: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建声明 - 证据边
        
        Args:
            num_evidences: 有效证据数量
            
        Returns:
            edge_index: (2, num_edges) 边索引 (0=claim, 1~N=evidence)
            edge_weight: (num_edges,) 边权重
        """
        if num_evidences == 0:
            return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
        
        # 声明节点索引为 0，证据节点索引为 1~num_evidences
        claim_idx = 0
        evidence_indices = np.arange(1, num_evidences + 1)
        
        # 双向边
        edge_index = np.concatenate([
            np.stack([np.full(num_evidences, claim_idx), evidence_indices], axis=0),
            np.stack([evidence_indices, np.full(num_evidences, claim_idx)], axis=0)
        ], axis=1)
        
        # 边权重初始化为 1
        edge_weights = np.ones(2 * num_evidences, dtype=np.float32)
        
        return edge_index, edge_weights
    
    def build_sample_graph(
        self,
        claim_emb: np.ndarray,
        evidence_emb: np.ndarray,
        evidence_mask: np.ndarray,
        retrieved_mask: np.ndarray = None,
        label: int = None
    ) -> Dict:
        """
        构建单个样本的图数据
        
        Args:
            claim_emb: (emb_dim,) 声明嵌入
            evidence_emb: (max_evidences, emb_dim) 证据嵌入
            evidence_mask: (max_evidences,) 证据掩码
            retrieved_mask: (max_evidences,) 检索证据掩码 (1=检索过，0=原始)
            label: 标签
            
        Returns:
            graph_data: 图数据字典
        """
        # 节点特征：[claim; evidence1; evidence2; ...]
        # 节点 0 是声明，节点 1~N 是证据
        num_valid_evidences = int(np.sum(evidence_mask > 0.5))
        num_nodes = 1 + num_valid_evidences
        
        # 节点特征矩阵
        node_features = np.zeros((num_nodes, self.emb_dim), dtype=np.float32)
        node_features[0] = claim_emb  # 声明节点
        
        # 证据节点 (只取有效证据)
        valid_evidence_indices = np.where(evidence_mask > 0.5)[0]
        for i, idx in enumerate(valid_evidence_indices):
            node_features[i + 1] = evidence_emb[idx]
        
        # 节点类型：0=claim, 1=evidence
        node_types = np.zeros(num_nodes, dtype=np.int64)
        node_types[1:] = 1  # 证据节点
        
        # 检索证据标记 (用于门控机制)
        # 只有证据节点有检索标记，声明节点为 -1
        retrieval_flags = np.full(num_nodes, -1, dtype=np.float32)
        if retrieved_mask is not None:
            for i, idx in enumerate(valid_evidence_indices):
                retrieval_flags[i + 1] = retrieved_mask[idx]
        
        # 构建边
        # 1. 声明 - 证据边
        ce_edge_index, ce_edge_weight = self.build_claim_evidence_edges(num_valid_evidences)
        
        # 2. 证据 - 证据边 (核心创新)
        ee_edge_index, ee_edge_weight = self.build_evidence_evidence_edges(
            evidence_emb, evidence_mask
        )
        
        # 调整证据 - 证据边的索引 (偏移 +1，因为节点 0 是声明)
        if ee_edge_index.shape[1] > 0:
            ee_edge_index = ee_edge_index + 1
        
        # 3. 自环 (帮助信息传播)
        self_loop_index = np.arange(num_nodes).reshape(1, -1)
        self_loop_index = np.concatenate([self_loop_index, self_loop_index], axis=0)
        self_loop_weight = np.ones(num_nodes, dtype=np.float32)
        
        # 合并所有边
        if ce_edge_index.shape[1] > 0 and ee_edge_index.shape[1] > 0:
            edge_index = np.concatenate([ce_edge_index, ee_edge_index, self_loop_index], axis=1)
            edge_weight = np.concatenate([ce_edge_weight, ee_edge_weight, self_loop_weight], axis=0)
            edge_type = np.concatenate([
                np.zeros(ce_edge_index.shape[1], dtype=np.int64),  # C-E 边
                np.ones(ee_edge_index.shape[1], dtype=np.int64),  # E-E 边
                np.full(self_loop_index.shape[1], 2, dtype=np.int64)  # 自环
            ])
        elif ce_edge_index.shape[1] > 0:
            edge_index = np.concatenate([ce_edge_index, self_loop_index], axis=1)
            edge_weight = np.concatenate([ce_edge_weight, self_loop_weight], axis=0)
            edge_type = np.concatenate([
                np.zeros(ce_edge_index.shape[1], dtype=np.int64),
                np.full(self_loop_index.shape[1], 2, dtype=np.int64)
            ])
        else:
            edge_index = self_loop_index
            edge_weight = self_loop_weight
            edge_type = np.full(self_loop_index.shape[1], 2, dtype=np.int64)
        
        # 构建图数据
        graph_data = {
            "node_features": torch.from_numpy(node_features),  # (num_nodes, emb_dim)
            "node_types": torch.from_numpy(node_types),  # (num_nodes,)
            "retrieval_flags": torch.from_numpy(retrieval_flags),  # (num_nodes,)
            "edge_index": torch.from_numpy(edge_index),  # (2, num_edges)
            "edge_weight": torch.from_numpy(edge_weight),  # (num_edges,)
            "edge_type": torch.from_numpy(edge_type),  # (num_edges,)
            "claim_node_idx": 0,
            "num_nodes": num_nodes,
            "num_evidences": num_valid_evidences
        }
        
        if label is not None:
            graph_data["label"] = torch.tensor(label, dtype=torch.long)
        
        return graph_data
    
    def build_dataset_graphs(
        self,
        claims_emb: np.ndarray,
        evidences_emb: np.ndarray,
        evidence_mask: np.ndarray,
        retrieved_mask: np.ndarray = None,
        labels: np.ndarray = None
    ) -> List[Dict]:
        """
        构建整个数据集的图
        
        Args:
            claims_emb: (N, emb_dim)
            evidences_emb: (N, max_evidences, emb_dim)
            evidence_mask: (N, max_evidences)
            retrieved_mask: (N, max_evidences) 可选
            labels: (N,) 可选
            
        Returns:
            graphs: List[Dict] 图数据列表
        """
        graphs = []
        num_samples = len(claims_emb)
        
        for i in range(num_samples):
            graph = self.build_sample_graph(
                claim_emb=claims_emb[i],
                evidence_emb=evidences_emb[i],
                evidence_mask=evidence_mask[i],
                retrieved_mask=retrieved_mask[i] if retrieved_mask is not None else None,
                label=labels[i] if labels is not None else None
            )
            graphs.append(graph)
            
            if (i + 1) % 100 == 0:
                print(f"  已构建 {i + 1}/{num_samples} 个图")
        
        return graphs
    
    def save_graphs(self, graphs: List[Dict], filepath: str):
        """保存图数据"""
        torch.save(graphs, filepath)
        print(f"已保存 {len(graphs)} 个图到 {filepath}")
    
    def load_graphs(self, filepath: str) -> List[Dict]:
        """加载图数据"""
        return torch.load(filepath)


def build_and_cache_graphs(config: Config = None):
    """
    主函数：加载数据并构建图缓存
    """
    if config is None:
        config = get_config()
    
    config.ensure_dirs()
    builder = GraphBuilder(config)
    
    print("=" * 60)
    print("开始构建异构图数据")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/4] 加载嵌入数据...")
    claims_path = config.path.get_full_path(config.path.claims_emb_file)
    
    # 在这里切换读取路径
    # evidences_path = config.path.get_full_path(config.path.evidences_prev_file)
    # mask_path = config.path.get_full_path(config.path.evd_mask_file)
    evidences_path = config.path.get_full_path(config.path.evidences_retrieved_file)
    mask_path = config.path.get_full_path(config.path.retrieved_mask_file)
    
    labels_path = config.path.get_full_path(config.path.labels_file)
    
    claims_emb = np.load(claims_path)
    evidences_emb = np.load(evidences_path)
    evidence_mask = np.load(mask_path)
    labels = np.load(labels_path)
    
    print(f"  声明嵌入：{claims_emb.shape}")
    print(f"  证据嵌入：{evidences_emb.shape}")
    print(f"  证据掩码：{evidence_mask.shape}")
    print(f"  标签：{labels.shape}")
    
    # 2. 加载检索证据标记 (如果存在)
    retrieved_mask = None
    retrieved_mask_path = config.path.get_full_path(config.path.retrieved_mask_file)
    if os.path.exists(retrieved_mask_path):
        print("\n[2/4] 加载检索证据标记...")
        retrieved_mask = np.load(retrieved_mask_path)
        print(f"  检索掩码：{retrieved_mask.shape}")
        print(f"  被置换证据比例：{retrieved_mask.mean():.2%}")
    else:
        print("\n[2/4] 未找到检索证据标记文件，使用全 0 掩码")
        retrieved_mask = np.zeros_like(evidence_mask)
    
    # 3. 构建图
    print("\n[3/4] 构建异构图...")
    graphs = builder.build_dataset_graphs(
        claims_emb=claims_emb,
        evidences_emb=evidences_emb,
        evidence_mask=evidence_mask,
        retrieved_mask=retrieved_mask,
        labels=labels
    )
    
    # 4. 保存图缓存
    print("\n[4/4] 保存图缓存...")
    graph_path = config.path.get_full_path(config.path.train_graph_file, is_graph_cache=True)
    builder.save_graphs(graphs, graph_path)
    
    # 统计信息
    print("\n" + "=" * 60)
    print("图构建完成 - 统计信息")
    print("=" * 60)
    avg_nodes = np.mean([g["num_nodes"] for g in graphs])
    avg_edges = np.mean([g["edge_index"].shape[1] for g in graphs])
    avg_evidences = np.mean([g["num_evidences"] for g in graphs])
    print(f"  样本总数：{len(graphs)}")
    print(f"  平均节点数：{avg_nodes:.2f}")
    print(f"  平均边数：{avg_edges:.2f}")
    print(f"  平均证据数：{avg_evidences:.2f}")
    print(f"  标签分布：{np.bincount(labels)}")
    print("=" * 60)
    
    return graphs


if __name__ == "__main__":
    build_and_cache_graphs()