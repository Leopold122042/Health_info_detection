"""
dataset_graph.py - 图数据集加载器
功能：加载预构建的图数据，支持 batching 和 collate
替代 baseline.py 中的 FactCheckDataset
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
from config import get_config, Config


class GraphFactCheckDataset(Dataset):
    """
    图结构事实核查数据集
    加载预构建的异构图数据
    """
    
    def __init__(
        self, 
        graph_file: str,
        config: Config = None
    ):
        """
        Args:
            graph_file: 图数据文件路径
            config: 配置对象
        """
        self.config = config if config else get_config()
        self.graph_file = graph_file
        
        print(f"[GraphDataset] 加载图数据：{graph_file}")
        self.graphs = torch.load(graph_file)
        print(f"[GraphDataset] 已加载 {len(self.graphs)} 个样本")
        
        # 统计信息
        self._compute_statistics()
    
    def _compute_statistics(self):
        """计算数据集统计信息"""
        if len(self.graphs) == 0:
            return
        
        self.num_nodes_list = [g["num_nodes"] for g in self.graphs]
        self.num_edges_list = [g["edge_index"].shape[1] for g in self.graphs]
        self.num_evidences_list = [g["num_evidences"] for g in self.graphs]
        
        # 标签分布
        if "label" in self.graphs[0]:
            labels = [g["label"].item() for g in self.graphs]
            self.label_distribution = np.bincount(labels)
        else:
            self.label_distribution = None
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个样本
        
        Returns:
            dict: 包含图数据的字典
        """
        graph = self.graphs[idx]
        
        item = {
            "node_features": graph["node_features"],  # (num_nodes, emb_dim)
            "node_types": graph["node_types"],  # (num_nodes,)
            "retrieval_flags": graph["retrieval_flags"],  # (num_nodes,)
            "edge_index": graph["edge_index"],  # (2, num_edges)
            "edge_weight": graph["edge_weight"],  # (num_edges,)
            "edge_type": graph["edge_type"],  # (num_edges,)
            "claim_node_idx": graph["claim_node_idx"],
            "num_nodes": graph["num_nodes"],
            "num_evidences": graph["num_evidences"]
        }
        
        if "label" in graph:
            item["label"] = graph["label"]
        
        return item
    
    def get_label_distribution(self) -> Optional[np.ndarray]:
        """获取标签分布"""
        return self.label_distribution
    
    def print_statistics(self):
        """打印数据集统计信息"""
        print("\n" + "=" * 60)
        print("数据集统计信息")
        print("=" * 60)
        print(f"  样本总数：{len(self)}")
        print(f"  平均节点数：{np.mean(self.num_nodes_list):.2f} ± {np.std(self.num_nodes_list):.2f}")
        print(f"  平均边数：{np.mean(self.num_edges_list):.2f} ± {np.std(self.num_edges_list):.2f}")
        print(f"  平均证据数：{np.mean(self.num_evidences_list):.2f} ± {np.std(self.num_evidences_list):.2f}")
        
        if self.label_distribution is not None:
            print(f"  标签分布：{self.label_distribution}")
            print(f"  类别不平衡比：{self.label_distribution.max() / self.label_distribution.min():.2f}")
        print("=" * 60)


class GraphCollator:
    """
    图数据批处理_collate_fn
    由于图大小不一，需要特殊处理 batching
    
    策略：每个 batch 中的图独立处理，不合并节点
    """
    
    def __init__(self, config: Config = None):
        self.config = config if config else get_config()
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        将多个图样本合并为 batch
        
        注意：由于图大小不一，我们返回样本列表而非合并的张量
        模型需要支持逐样本处理或实现图批处理
        
        Args:
            batch: List[Dict] 样本列表
            
        Returns:
            batch_data: 批处理后的数据
        """
        # 方案 1：返回样本列表 (简单，适合小 batch)
        # 方案 2：合并所有图为一个大图 (复杂，需要 offset 索引)
        
        # 这里采用方案 1，保持简单
        # 后续模型需要支持循环处理 batch 中的每个样本
        
        batch_data = {
            "graphs": batch,
            "batch_size": len(batch)
        }
        
        # 如果有标签，单独提取
        if "label" in batch[0]:
            batch_data["labels"] = torch.stack([item["label"] for item in batch], dim=0)
        
        return batch_data


class BatchedGraphCollator:
    """
    批处理图_collate_fn (合并所有图为一个大图)
    适合 PyG 风格的 GNN 处理
    """
    
    def __init__(self, config: Config = None):
        self.config = config if config else get_config()
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        合并多个图为一个大图，通过 offset 索引区分
        
        Returns:
            batched_graph: 合并后的图数据
        """
        batch_size = len(batch)
        
        # 累计节点数用于 offset
        node_offset = 0
        all_node_features = []
        all_node_types = []
        all_retrieval_flags = []
        all_edge_indices = []
        all_edge_weights = []
        all_edge_types = []
        all_labels = []
        claim_node_indices = []
        
        for i, graph in enumerate(batch):
            num_nodes = graph["num_nodes"]
            
            # 节点特征
            all_node_features.append(graph["node_features"])
            all_node_types.append(graph["node_types"])
            all_retrieval_flags.append(graph["retrieval_flags"])
            
            # 边索引 (添加 offset)
            edge_index = graph["edge_index"] + node_offset  # (2, num_edges)
            all_edge_indices.append(edge_index)
            
            # 边权重和类型
            all_edge_weights.append(graph["edge_weight"])
            all_edge_types.append(graph["edge_type"])
            
            # 标签
            if "label" in graph:
                all_labels.append(graph["label"])
            
            # 声明节点索引 (用于最终分类)
            claim_node_indices.append(graph["claim_node_idx"] + node_offset)
            
            # 更新 offset
            node_offset += num_nodes
        
        # 合并所有张量
        batched_data = {
            "node_features": torch.cat(all_node_features, dim=0),  # (total_nodes, emb_dim)
            "node_types": torch.cat(all_node_types, dim=0),  # (total_nodes,)
            "retrieval_flags": torch.cat(all_retrieval_flags, dim=0),  # (total_nodes,)
            "edge_index": torch.cat(all_edge_indices, dim=1),  # (2, total_edges)
            "edge_weight": torch.cat(all_edge_weights, dim=0),  # (total_edges,)
            "edge_type": torch.cat(all_edge_types, dim=0),  # (total_edges,)
            "claim_node_indices": torch.tensor(claim_node_indices, dtype=torch.long),  # (batch_size,)
            "batch_size": batch_size,
            "num_graphs": batch_size
        }
        
        if len(all_labels) > 0:
            batched_data["labels"] = torch.stack(all_labels, dim=0)  # (batch_size,)
        
        return batched_data


def create_dataloaders(
    config: Config = None,
    use_batched_collator: bool = True
) -> tuple:
    """
    创建训练、验证、测试数据加载器
    
    Args:
        config: 配置对象
        use_batched_collator: 是否使用批处理_collator (推荐 True)
        
    Returns:
        train_loader, val_loader, test_loader
    """
    if config is None:
        config = get_config()
    
    config.ensure_dirs()
    
    # 加载图文件
    train_file = config.path.get_full_path(config.path.train_graph_file, is_graph_cache=True)
    
    # 注意：实际使用时需要划分 train/val/test
    # 这里简化为全部作为训练集，实际应使用 split
    dataset = GraphFactCheckDataset(train_file, config)
    dataset.print_statistics()
    
    # 选择 collator
    if use_batched_collator:
        collator = BatchedGraphCollator(config)
        print("[DataLoader] 使用 BatchedGraphCollator (合并图)")
    else:
        collator = GraphCollator(config)
        print("[DataLoader] 使用 GraphCollator (独立图)")
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # 图数据加载复杂，建议 0
        pin_memory=True
    )
    
    # val 和 test loader 需要单独的图文件
    # 这里返回 None，实际使用时需要构建
    val_loader = None
    test_loader = None
    
    return train_loader, val_loader, test_loader, dataset


if __name__ == "__main__":
    # 测试数据集加载
    config = get_config()
    config.ensure_dirs()
    
    # 先构建图 (如果还没构建)
    import os
    train_graph_path = config.path.get_full_path(config.path.train_graph_file, is_graph_cache=True)
    
    if not os.path.exists(train_graph_path):
        print("图文件不存在，先构建图...")
        from graph_builder import build_and_cache_graphs
        build_and_cache_graphs(config)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, dataset = create_dataloaders(config)
    
    # 测试 batch 加载
    print("\n测试 batch 加载...")
    for i, batch in enumerate(train_loader):
        print(f"\nBatch {i + 1}:")
        print(f"  节点特征形状：{batch['node_features'].shape}")
        print(f"  边索引形状：{batch['edge_index'].shape}")
        print(f"  标签形状：{batch['labels'].shape if 'labels' in batch else 'N/A'}")
        print(f"  声明节点索引：{batch['claim_node_indices']}")
        
        if i >= 2:  # 只测试前 3 个 batch
            break
    
    print("\n数据加载器测试完成!")