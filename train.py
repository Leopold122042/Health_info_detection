"""
train.py - 完整训练循环
功能：实现早停、梯度裁剪、模型保存、加权损失
"""

import torch
import torch.nn as nn
import numpy as np
import os
import time
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import get_config, Config
from dataset_graph import create_dataloaders, GraphFactCheckDataset, BatchedGraphCollator
from models_gnn import CA_HGER_Model
from loss_functions import CombinedLoss
from metrics import compute_metrics, print_classification_report
from torch.utils.data import random_split, DataLoader


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=5, min_delta=0.001, path='checkpoints/best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_macro_f1 = 0
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
    def __call__(self, macro_f1, model):
        score = macro_f1
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score - self.min_delta:
            self.counter += 1
            print(f"  早停计数：{self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_macro_f1 = score
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        """保存最佳模型"""
        torch.save(model.state_dict(), self.path)
        print(f"  ✓ 模型已保存至 {self.path} (Macro-F1: {self.best_macro_f1:.4f})")


def calculate_class_weights(labels):
    """计算类别权重以解决不平衡"""
    from collections import Counter
    counter = Counter(labels)
    total = sum(counter.values())
    # 权重 = 总样本数 / (类别数 * 该类样本数)
    weights = [total / (len(counter) * counter[i]) for i in range(len(counter))]
    return weights


def train_one_epoch(model, loader, optimizer, criterion, device, config):
    """单次 epoch 训练"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_cons_loss = 0
    num_batches = 0
    
    for batch in loader:
        # 数据移至设备
        batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # 前向传播
        logits, consistency_loss, _ = model(batch_data)
        labels = batch_data['labels']
        
        # 计算损失
        total_loss_batch, ce_loss_batch, cons_loss_batch = criterion(logits, labels, consistency_loss)
        
        # 反向传播
        total_loss_batch.backward()
        
        # 梯度裁剪 (防止 GNN 梯度爆炸)
        if hasattr(config.train, 'gradient_clip_val') and config.train.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_val)
        
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        total_ce_loss += ce_loss_batch.item()
        total_cons_loss += cons_loss_batch.item()
        num_batches += 1
        
    return total_loss / num_batches, total_ce_loss / num_batches, total_cons_loss / num_batches


def evaluate(model, loader, criterion, device, config):
    """验证/测试评估"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_golds = []
    all_probs = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            logits, consistency_loss, _ = model(batch_data)
            labels = batch_data['labels']
            
            # 损失
            loss, _, _ = criterion(logits, labels, consistency_loss)
            total_loss += loss.item()
            
            # 预测
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_golds.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            num_batches += 1
            
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    all_probs = np.array(all_probs)
    
    # 计算指标
    metrics = compute_metrics(all_golds, all_preds, all_probs, num_labels=config.model.num_labels if hasattr(config.model, 'num_labels') else 2)
    
    return avg_loss, metrics, all_preds, all_golds


def run_training(config=None):
    """主训练流程"""
    if config is None:
        config = get_config()
    
    config.ensure_dirs()
    device = config.device
    print(f"=== 开始训练 (Device: {device}) ===")
    
    # 1. 数据准备 (划分 train/val)
    print("\n[1/5] 加载数据...")
    # 注意：Phase 1 只生成了 train_graphs.pt，这里需要手动划分或生成 val 文件
    # 为简化，我们加载全部数据并在内存中划分
    from dataset_graph import GraphFactCheckDataset
    train_graph_path = config.path.get_full_path(config.path.train_graph_file, is_graph_cache=True)
    dataset = GraphFactCheckDataset(train_graph_path, config)
    
    # 获取标签用于计算权重
    all_labels = [g['label'].item() for g in dataset.graphs]
    class_weights = calculate_class_weights(all_labels) if config.train.use_weighted_loss else None
    print(f"  类别权重：{class_weights}")
    
    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.train.seed))
    
    collator = BatchedGraphCollator(config)
    train_loader = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True, collate_fn=collator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.train.batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    
    # 2. 模型初始化
    print("\n[2/5] 初始化模型...")
    model = CA_HGER_Model(config).to(device)
    
    # 3. 优化器与损失
    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    criterion = CombinedLoss(class_weights=class_weights, config=config)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=config.train.early_stopping_patience, path='checkpoints/best_model.pth')
    
    # 4. 训练循环
    print("\n[3/5] 开始训练...")
    best_val_macro_f1 = 0
    history = {'train_loss': [], 'val_loss': [], 'val_macro_f1': []}
    
    for epoch in range(config.train.epochs):
        start_time = time.time()
        
        # 训练
        train_loss, train_ce, train_cons = train_one_epoch(model, train_loader, optimizer, criterion, device, config)
        
        # 验证
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device, config)
        
        # 记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        
        # 日志
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.train.epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f} (CE: {train_ce:.4f}, Cons: {train_cons:.4f})")
        print(f"  Val Loss: {val_loss:.4f} | Macro-F1: {val_metrics['macro_f1']:.4f} | MCC: {val_metrics['mcc']:.4f}")
        
        # 学习率调整
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['macro_f1'])
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"  → 学习率调整：{old_lr:.6f} → {new_lr:.6f}")
        
        # 早停检查
        early_stopping(val_metrics['macro_f1'], model)
        if early_stopping.early_stop:
            print("\n  ⚠ 触发早停，训练结束")
            break
            
        if val_metrics['macro_f1'] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics['macro_f1']
    
    # 5. 最终评估
    print("\n[4/5] 加载最佳模型进行最终评估...")
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
    _, test_metrics, preds, golds = evaluate(model, val_loader, criterion, device, config)
    
    print("\n" + "="*60)
    print("最终验证集结果 (Best Model)")
    print("="*60)
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Micro-F1: {test_metrics['micro_f1']:.4f}")
    print(f"  MCC:      {test_metrics['mcc']:.4f}")
    print(f"  AUC:      {test_metrics['auc']:.4f}")
    print("="*60)
    
    print_classification_report(golds, preds)
    
    return history, test_metrics


if __name__ == "__main__":
    config = get_config()
    # 确保配置中有 num_labels
    if not hasattr(config.model, 'num_labels'):
        config.model.num_labels = 2
    run_training(config)