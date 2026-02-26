import json
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

def generate_strict_features_v2(json_path, output_dir="cache", model_name='bert-base-chinese'):
    # 1. 初始化路径与目录
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True) # 自动创建 cache 目录
    
    # 2. 读取检索后的 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    num_claims = len(data)
    num_evidences = 5
    emb_dim = 768
    
    # 3. 预分配张量空间，确保物理隔离
    all_labels = np.zeros(num_claims, dtype=int)
    all_evd_embeddings = np.zeros((num_claims, num_evidences, emb_dim), dtype=np.float32)
    all_evd_masks = np.zeros((num_claims, num_evidences), dtype=int)

    valid_texts = []
    positions = []

    # 提取内容并生成掩码逻辑
    for i, item in enumerate(data):
        # 标签分类：0-真实, 1-虚假, 2-不确定
        all_labels[i] = item.get('label', 2) 
        evid_dict = item.get('evidence', {})
        
        for slot in range(num_evidences):
            # 获取检索到的证据内容
            content = evid_dict.get(str(slot), {}).get('content', "").strip()
            if content:
                valid_texts.append(content)
                positions.append((i, slot))
                all_evd_masks[i, slot] = 1 # 激活掩码：该位置有证据
            else:
                all_evd_masks[i, slot] = 0 # 忽略空行

    # 4. 执行批量 Embedding 编码
    print(f"[Sigma] Encoding {len(valid_texts)} valid evidences into {output_dir}...")
    model = SentenceTransformer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    embeddings = model.encode(
        valid_texts, 
        batch_size=64, 
        show_progress_bar=True, 
        normalize_embeddings=True
    )

    # 5. 回填至三维张量 (N, 5, 768)
    for idx, (c_idx, e_idx) in enumerate(positions):
        all_evd_embeddings[c_idx, e_idx] = embeddings[idx]

    # 6. 存储至指定缓存目录
    np.save(out_path / 'labels.npy', all_labels)
    np.save(out_path / 'evidences_embeddings_prev.npy', all_evd_embeddings)
    np.save(out_path / 'evd_mask.npy', all_evd_masks)
    
    print(f"Success! All features saved to {out_path.absolute()}")

if __name__ == "__main__":
    # 假设你的检索结果文件在这个路径
    INPUT_JSON = "data/health_info.json" 
    generate_strict_features_v2(INPUT_JSON)