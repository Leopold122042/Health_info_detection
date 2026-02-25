"""
metrics.py - 详细评估指标计算
功能：计算 Macro-F1, MCC, AUC 等 (Feng et al. 2025 推荐)
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, matthews_corrcoef, roc_auc_score, classification_report
import torch


def compute_metrics(y_true, y_pred, y_prob=None, num_labels=2):
    """
    计算全面评估指标
    
    Args:
        y_true: list/array 真实标签
        y_pred: list/array 预测标签
        y_prob: list/array 预测概率 (用于 AUC)
        num_labels: 类别数
        
    Returns:
        dict: 指标字典
    """
    metrics = {}
    
    # 1. 基础指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # 2. F1 分数 (核心指标)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro')
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # 3. 精确率与召回率 (Macro 平均)
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    metrics['macro_precision'] = precision
    metrics['macro_recall'] = recall
    
    # 4. MCC (Feng et al. 2025 推荐用于不平衡数据)
    try:
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    except Exception:
        metrics['mcc'] = 0.0
        
    # 5. AUC (二分类特有)
    if y_prob is not None and num_labels == 2:
        try:
            # 取正类的概率
            if len(y_prob.shape) == 1:
                prob_pos = y_prob
            else:
                prob_pos = y_prob[:, 1]
            metrics['auc'] = roc_auc_score(y_true, prob_pos)
        except Exception:
            metrics['auc'] = 0.5
    else:
        metrics['auc'] = 0.5
        
    return metrics


def print_classification_report(y_true, y_pred, target_names=None):
    """打印详细分类报告"""
    if target_names is None:
        target_names = ['True', 'Fake']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\n" + "="*60)
    print("分类详细报告")
    print("="*60)
    print(report)
    print("="*60)