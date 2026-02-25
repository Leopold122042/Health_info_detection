"""
loss_functions.py - 自定义损失函数
功能：实现加权交叉熵损失、一致性损失融合
创新点：解决类别不平衡 (Feng et al. 2025 建议)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import get_config


class WeightedCrossEntropyLoss(nn.Module):
    """
    加权交叉熵损失
    功能：根据类别分布自动计算权重，解决类别不平衡问题
    """
    def __init__(self, class_weights=None, config=None):
        super().__init__()
        self.config = config if config else get_config()
        self.class_weights = class_weights
        
    def forward(self, logits, labels):
        """
        Args:
            logits: (batch_size, num_labels)
            labels: (batch_size,)
        """
        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, device=logits.device, dtype=torch.float32)
            loss = F.cross_entropy(logits, labels, weight=weights)
        else:
            loss = F.cross_entropy(logits, labels)
        return loss


class ConsistencyLossWrapper(nn.Module):
    """
    一致性损失包装器
    功能：将模型内部计算的一致性损失纳入总损失
    对应文献 1：利用事件级/证据级一致性辅助文档级检测
    """
    def __init__(self, weight=0.1, config=None):
        super().__init__()
        self.config = config if config else get_config()
        self.weight = weight
        
    def forward(self, consistency_loss):
        """
        Args:
            consistency_loss: scalar (来自模型输出)
        """
        return self.weight * consistency_loss


class CombinedLoss(nn.Module):
    """
    组合损失函数
    总损失 = CE_Loss + λ * Consistency_Loss
    """
    def __init__(self, class_weights=None, config=None):
        super().__init__()
        self.config = config if config else get_config()
        
        # 分类损失
        self.ce_loss = WeightedCrossEntropyLoss(class_weights=class_weights, config=config)
        
        # 一致性损失权重
        self.consistency_weight = self.config.train.consistency_loss_weight if hasattr(self.config.train, 'consistency_loss_weight') else 0.1
        self.consistency_loss_wrapper = ConsistencyLossWrapper(weight=self.consistency_weight, config=config)
        
    def forward(self, logits, labels, consistency_loss):
        """
        Args:
            logits: (batch_size, num_labels)
            labels: (batch_size,)
            consistency_loss: scalar (来自模型)
        """
        # 1. 分类损失
        loss_ce = self.ce_loss(logits, labels)
        
        # 2. 一致性损失
        loss_cons = self.consistency_loss_wrapper(consistency_loss)
        
        # 3. 总损失
        total_loss = loss_ce + loss_cons
        
        return total_loss, loss_ce, loss_cons