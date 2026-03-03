import pandas as pd
from train import run_kfold # 复用之前的 K-折训练逻辑
from pathlib import Path

def run_ablation():
    configs = [
        {"name": "Full Model", "params": {"use_nli": True, "use_tfidf": True, "use_ee": True}},
        {"name": "w/o NLI",    "params": {"use_nli": False, "use_tfidf": True, "use_ee": True}},
        {"name": "w/o TF-IDF", "params": {"use_nli": True, "use_tfidf": False, "use_ee": True}},
        {"name": "w/o EE-Int", "params": {"use_nli": True, "use_tfidf": True, "use_ee": False}},
    ]
    
    summary_results = []

    for config in configs:
        print(f"\n运行消融实验组: {config['name']}")
        # 注意：你需要微调 run_kfold 使其接受 model_params 参数
        avg_metrics = run_kfold(k=5, epochs=20, model_params=config['params'])
        
        summary_results.append({
            "Variant": config['name'],
            "Acc": avg_metrics['acc'],
            "F1": avg_metrics['f1'],
            "MCC": avg_metrics['mcc']
        })

    # 导出对比表格
    df = pd.DataFrame(summary_results)
    Path("outputs/ablation").mkdir(parents=True, exist_ok=True)
    df.to_csv("outputs/ablation/ablation_summary.csv", index=False)
    print("\n消融实验汇总完成，结果已保存至 outputs/ablation/ablation_summary.csv")

if __name__ == "__main__":
    run_ablation()