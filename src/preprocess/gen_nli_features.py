import json
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def gen_nli_features(json_path, output_dir="cache", model_name='IDEA-CCNL/Erlangshen-Roberta-110M-NLI'):
    # 1. 配置设备与加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在加载模型: {model_name}，设备: {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # 获取模型标签维度 (该模型应为 3: Entailment, Neutral, Contradiction)
    out_dim = model.config.num_labels
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v.upper(): int(k) for k, v in id2label.items()}
    idx_contra = label2id.get("CONTRADICTION", 0)
    idx_neutral = label2id.get("NEUTRAL", 1)
    idx_entail = label2id.get("ENTAILMENT", 2)
    print(f"模型检测完成，输出维度为: {out_dim}")

    # 2. 读取数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    N = len(data)
    num_evd = 5
    
    # 预分配空间
    nli_ce = np.zeros((N, num_evd, out_dim), dtype=np.float32)
    nli_ee = np.zeros((N, num_evd, num_evd, out_dim), dtype=np.float32)

    # 3. 开始处理
    for i, item in enumerate(tqdm(data, desc="提取 NLI 特征")):
        claim = item['claim']
        # 获取 5 条证据内容
        evidences = []
        for j in range(num_evd):
            content = item['evidence'].get(str(j), {}).get('content', "").strip()
            evidences.append(content)
        
        # --- A. 计算 Claim-Evidence (CE) 逻辑关系 ---
        for j in range(num_evd):
            if not evidences[j]:
                # 空证据设为“中立”概率为 1.0 (Index 1)
                fill_val = [0.0] * out_dim
                fill_val[idx_neutral] = 1.0 
                nli_ce[i, j] = fill_val
                continue
            
            # 使用 Roberta 推荐的输入格式
            inputs = tokenizer(claim, evidences[j], return_tensors='pt', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                nli_ce[i, j] = probs

        # --- B. 计算 Evidence-Evidence (EE) 逻辑关系 (核心创新) ---
        for j in range(num_evd):
            for k in range(num_evd):
                if j == k:
                    # 相同证据设为“蕴含”概率为 1.0 (Index 0)
                    fill_val = [0.0] * out_dim
                    fill_val[idx_entail] = 1.0
                    nli_ee[i, j, k] = fill_val
                    continue
                    
                if not evidences[j] or not evidences[k]:
                    # 只要有一个为空，设为中立
                    fill_val = [0.0] * out_dim
                    fill_val[idx_neutral] = 1.0 
                    nli_ee[i, j, k] = fill_val
                    continue
                
                # 计算两两之间的逻辑
                inputs = tokenizer(evidences[j], evidences[k], return_tensors='pt', truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    nli_ee[i, j, k] = probs

    # 4. 保存文件
    Path(output_dir).mkdir(exist_ok=True)
    np.save(f"{output_dir}/nli_logits_ce.npy", nli_ce)
    np.save(f"{output_dir}/nli_logits_ee.npy", nli_ee)
    print(f"成功保存特征至 {output_dir}。文件尺寸: CE {nli_ce.shape}, EE {nli_ee.shape}")

if __name__ == "__main__":
    # 请确保路径指向你优化后的 JSON 文件
    DATA_PATH = "data/health_info_retrieved.json"
    gen_nli_features(DATA_PATH)
