"""
基于 run_feasibility_fixed.py 的精简优化版本：
1. 保留 Milvus 向量检索与证据复用统计所需的核心逻辑
2. 完全删除所有绘图 / 可视化相关代码
3. 新增匹配插入逻辑：
   - 若原证据 = 5条：保留相似度最高的3条，替换相似度最低的2条（需检索相似度 > 0.7）
   - 若原证据 < 5条：保留全部原证据，空位按相似度由高到低补齐（需检索相似度 > 0.7）
4. 输出 JSON 的结构与原 health_info.json 完全一致
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

from milvus_reuse_pipeline import EvidenceReusePipeline
from reuse_analysis import reuse_statistics, label_consistency


# ===============================
# Step 0: 数据读取
# ===============================

def load_health_info(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    claim_texts, claim_labels, evidence_texts = [], [], []
    for item in data:
        claim_texts.append(item["claim"])
        claim_labels.append(item["label"])
        for ev in item.get("evidence", {}).values():
            content = ev.get("content", "").strip()
            if content:
                evidence_texts.append(content)

    return data, claim_texts, claim_labels, evidence_texts


# ===============================
# Step 1: Embedding（带缓存）
# ===============================

def encode_with_cache(texts, model, cache_path, batch_size=32):
    if cache_path.exists():
        print(f"[Load] {cache_path.name}")
        return np.load(cache_path)
    print(f"[Compute] Encoding {cache_path.name}")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # 归一化后，内积(IP)即为余弦相似度
    )
    np.save(cache_path, emb)
    return emb


# ===============================
# Step 2: 根据新规则处理并另存配对
# ===============================

def process_and_save_pairs(original_data, claim_embeddings, evidence_embeddings, evidence_texts, search_results, output_path):
    """
    处理新的证据插入/替换逻辑，并保存为新的 JSON
    """
    new_data = []
    
    # 建立文本到向量、文本到 ID 的映射，用于快速计算原证据相似度及统计
    text_to_emb = {text: emb for text, emb in zip(evidence_texts, evidence_embeddings)}
    text_to_eid = {text: eid for eid, text in enumerate(evidence_texts)}
    
    evidence_to_claims = defaultdict(list)

    for cid, item in enumerate(original_data):
        new_item = dict(item) 
        claim_emb = claim_embeddings[cid]

        # 1. 提取当前 Claim 的所有原证据
        orig_evs = []
        for ev in item.get("evidence", {}).values():
            content = ev.get("content", "").strip()
            if content:
                orig_evs.append(content)
                
        # 2. 计算原证据与当前 Claim 的相似度得分
        orig_evs_with_scores = []
        for text in orig_evs:
            if text in text_to_emb:
                # 向量已归一化，使用 np.dot 计算余弦相似度
                score = np.dot(claim_emb, text_to_emb[text])
            else:
                score = 0.0
            orig_evs_with_scores.append((text, score))

        # 3. 核心逻辑：优胜劣汰 / 填补空缺
        if len(orig_evs_with_scores) >= 5:
            # 规则1：如果是5条，按相似度降序，保留前3，淘汰后2
            orig_evs_with_scores.sort(key=lambda x: x[1], reverse=True)
            kept_evs = [x[0] for x in orig_evs_with_scores[:3]]
            insert_capacity = 2
        elif len(orig_evs_with_scores) >= 4:
            orig_evs_with_scores.sort(key=lambda x: x[1], reverse=True)
            kept_evs = [x[0] for x in orig_evs_with_scores[:3]]
            insert_capacity = 1
        else:
            # 规则2：如果小于5条，保留全部原证据
            kept_evs = [x[0] for x in orig_evs_with_scores]
            insert_capacity = 5 - len(kept_evs)

        # 4. 从 Milvus 检索结果中筛选可插入的新证据
        retrieved_texts = []
        for hit in search_results[cid]:
            # 必须满足相似度 > 0.7
            if hit.distance > 0.7:
                text = evidence_texts[hit.id]
                # 确保不插入已经保留的原证据，也不重复插入
                if text not in kept_evs and text not in retrieved_texts:
                    retrieved_texts.append(text)
                    
        # 5. 组装最终证据（如果高分检索结果不足，能插几个插几个）
        final_evs = kept_evs + retrieved_texts[:insert_capacity]
        
        # 记录映射用于最后的统计分析
        for text in final_evs:
            eid = text_to_eid.get(text, -1)
            if eid != -1:
                evidence_to_claims[eid].append(cid)

        # 6. 格式化对齐原始 JSON 结构
        retrieved_evidence = {}
        for i, text in enumerate(final_evs):
            retrieved_evidence[str(i)] = {"content": text}
            
        new_item["evidence"] = retrieved_evidence
        new_data.append(new_item)

    # 保存文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"\n[Saved] Retrieved claim–evidence JSON -> {output_path}")
    return evidence_to_claims


# ===============================
# Step 3: 主流程
# ===============================

def main():
    data_path = Path("Health_misinfo_detection/health_info.json")
    output_path = Path("Health_misinfo_detection/health_info_retrieved.json")
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    raw_data, claim_texts, claim_labels, evidence_texts = load_health_info(data_path)
    print(f"#Claims: {len(claim_texts)}")
    print(f"#Evidences: {len(evidence_texts)}")

    encoder = SentenceTransformer("BAAI/bge-base-zh-v1.5")

    claim_embeddings = encode_with_cache(
        claim_texts, encoder, cache_dir / "claims_embeddings.npy"
    )
    evidence_embeddings = encode_with_cache(
        evidence_texts, encoder, cache_dir / "evidences_embeddings_prev.npy"
    )

    # ---- Milvus 向量检索 ----
    pipeline = EvidenceReusePipeline(dim=claim_embeddings.shape[1])
    pipeline.build_collection()
    pipeline.insert_evidences(evidence_embeddings)
    pipeline.build_index()

    # limit 放大到 10：考虑到前几个检索结果极有可能就是我们要保留的原证据，
    # 扩大 limit 可保证即使跳过原证据，依然有足够的 top 备选项。
    search_results = pipeline.collection.search(
        data=claim_embeddings.tolist(),
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 16}},
        limit=10, 
    )

    # ---- 处理规则并保存 JSON ----
    evidence_to_claims = process_and_save_pairs(
        original_data=raw_data,
        claim_embeddings=claim_embeddings,
        evidence_embeddings=evidence_embeddings,
        evidence_texts=evidence_texts,
        search_results=search_results,
        output_path=output_path,
    )

    # ---- 统计最终结果（基于更新后的映射关系） ----
    reuse_stats = reuse_statistics(evidence_to_claims)
    label_stats = label_consistency(evidence_to_claims, claim_labels)

    print("\n[Evidence Reuse Statistics]")
    for k, v in reuse_stats.items():
        print(f"{k}: {v}")

    print("\n[Label Consistency]")
    for k, v in label_stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()