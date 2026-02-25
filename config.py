"""
config.py - 全局配置管理
功能：集中管理所有超参数、路径配置和创新点参数
"""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PathConfig:
    """路径配置"""
    # 基础目录
    project_root: str = os.path.dirname(os.path.abspath(__file__))
    data_dir: str = "cache"
    graph_cache_dir: str = "graph_cache"
    
    # 输入文件
    claims_emb_file: str = "claims_embeddings.npy"
    evidences_prev_file: str = "evidences_embeddings_prev.npy"
    evidences_retrieved_file: str = "evidences_embeddings_r.npy"
    evd_mask_file: str = "evd_mask.npy"
    retrieved_mask_file: str = "evd_mask_r.npy" 
    labels_file: str = "labels.npy"
    
    # 输出文件
    train_graph_file: str = "train_graphs.pt"
    val_graph_file: str = "val_graphs.pt"
    test_graph_file: str = "test_graphs.pt"
    
    def get_full_path(self, filename: str, is_graph_cache: bool = False) -> str:
        """获取完整文件路径"""
        base_dir = self.graph_cache_dir if is_graph_cache else self.data_dir
        return os.path.join(self.project_root, base_dir, filename)


@dataclass
class ModelConfig:
    """模型配置"""
    # 嵌入维度
    emb_dim: int = 768  # BERT 嵌入维度
    hidden_dim: int = 256  # GNN 隐藏层维度
    gnn_layers: int = 3  # GNN 层数
    num_heads: int = 4  # 注意力头数
    
    # 图结构配置
    max_evidences: int = 5  # 最大证据数量
    claim_node_dim: int = 768
    evidence_node_dim: int = 768
    
    # 创新点：检索证据门控
    retrieval_gate_init: float = 0.5  # 门控初始值
    retrieval_gate_learnable: bool = True  # 门控是否可学习
    
    # 创新点：一致性阈值
    consistency_threshold: float = 0.7  # 证据间一致性阈值
    use_consistency_loss: bool = True  # 是否使用一致性损失


@dataclass
class GraphConfig:
    """图构建配置"""
    # 边类型
    edge_types: List[str] = None
    
    def __post_init__(self):
        if self.edge_types is None:
            self.edge_types = [
                "claim_to_evidence",      # 声明->证据
                "evidence_to_claim",      # 证据->声明
                "evidence_to_evidence",   # 证据->证据 (核心创新)
                "self_loop"               # 自环
            ]
    
    # 相似度计算
    similarity_metric: str = "cosine"  # cosine | euclidean | dot
    ee_edge_threshold: float = 0.5  # 证据 - 证据边相似度阈值
    
    # 节点类型
    node_types: List[str] = None
    
    def __post_init__(self):
        if self.node_types is None:
            self.node_types = ["claim", "evidence"]


@dataclass
class TrainConfig:
    """训练配置"""
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 5e-5
    weight_decay: float = 5e-5
    dropout: float = 0.5
    
     # [新增] 训练策略
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    gradient_clip_val: float = 1.0  # 梯度裁剪
    
    # [新增] 损失函数
    consistency_loss_weight: float = 0.1  # 一致性损失权重 λ
    use_weighted_loss: bool = True        # 是否启用加权 CE 解决不平衡
    
    # 优化器
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealing"
    
    # 随机种子
    seed: int = 42
    num_runs: int = 5
    
    # 数据集划分
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class Config:
    """总配置类"""
    path: PathConfig = None
    model: ModelConfig = None
    graph: GraphConfig = None
    train: TrainConfig = None
    
    def __post_init__(self):
        if self.path is None:
            self.path = PathConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.graph is None:
            self.graph = GraphConfig()
        if self.train is None:
            self.train = TrainConfig()
    
    # 设备配置
    @property
    def device(self):
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def ensure_dirs(self):
        """确保所有目录存在"""
        os.makedirs(self.path.data_dir, exist_ok=True)
        os.makedirs(self.path.graph_cache_dir, exist_ok=True)


# 全局配置实例
config = Config()


def get_config() -> Config:
    """获取配置实例"""
    return config


if __name__ == "__main__":
    # 测试配置
    cfg = get_config()
    cfg.ensure_dirs()
    print(f"项目根目录：{cfg.path.project_root}")
    print(f"数据目录：{cfg.path.data_dir}")
    print(f"图缓存目录：{cfg.path.graph_cache_dir}")
    print(f"设备：{cfg.device}")
    print(f"证据 - 证据边阈值：{cfg.graph.ee_edge_threshold}")
    print(f"检索门控初始值：{cfg.model.retrieval_gate_init}")