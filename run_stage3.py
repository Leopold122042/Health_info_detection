"""
run_stage3.py - 第三阶段启动脚本
功能：统一入口，检查依赖，启动训练
"""

import sys
import os

def check_dependencies():
    """检查必要文件是否存在"""
    required_files = [
        'config.py',
        'graph_builder.py',
        'dataset_graph.py',
        'models_gnn.py',
        'modules_consistency.py',
        'loss_functions.py',
        'metrics.py',
        'train.py'
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"✗ 错误：缺少以下文件：{missing}")
        print("  请确保第一阶段和第二阶段代码已生成。")
        return False
    return True

def check_data():
    """检查数据文件"""
    data_files = [
        'cache/claims_embeddings.npy',
        'cache/evidences_embeddings_r.npy',
        'cache/evd_mask.npy',
        'cache/evd_mask_r.npy',
        'cache/labels.npy',
        'graph_cache/train_graphs.pt'
    ]
    
    missing = [f for f in data_files if not os.path.exists(f)]
    if missing:
        print(f"⚠ 警告：缺少以下数据文件：{missing}")
        print("  请先运行 Phase 1 (graph_builder.py) 生成图数据。")
        return False
    return True

if __name__ == "__main__":
    print("="*60)
    print("健康虚假信息检测 - 第三阶段训练启动")
    print("="*60)
    
    if not check_dependencies():
        sys.exit(1)
        
    if not check_data():
        # 尝试自动生成图数据
        print("\n  尝试自动构建图数据...")
        from graph_builder import build_and_cache_graphs
        from config import get_config
        build_and_cache_graphs(get_config())
    
    print("\n  开始训练...")
    from train import run_training
    from config import get_config
    
    try:
        history, metrics = run_training(get_config())
        print("\n✓ 训练完成！")
    except Exception as e:
        print(f"\n✗ 训练失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)