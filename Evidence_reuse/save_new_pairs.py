"""
基于 run_feasibility_fixed.py 的精简版本：
1. 保留 Milvus 向量检索与证据复用统计所需的核心逻辑
2. 完全删除所有绘图 / 可视化相关代码
3. 新增：将「向量检索后的 claim–evidence 新配对结果」另存为 JSON
4. 输出 JSON 的结构与原 health_info.json 完全一致，仅替换 evidence 内容

该脚本用于：
- 固化向量检索结果
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
        normalize_embeddings=True,
    )
    np.save(cache_path, emb)
    return emb


# ===============================
# Step 2: 另存检索后的 claim–evidence 配对
# ===============================

def save_retrieved_pairs(original_data, evidence_texts, claim_to_hits, output_path, fallback_evidence):

    new_data = []

    for cid, item in enumerate(original_data):
        new_item = dict(item)  # 保留 claim / label 等字段

        retrieved_evidence = {}
        hits_ids = claim_to_hits.get(cid, [])
        source_list = [evidence_texts[eid] for eid in hits_ids] if hits_ids else fallback_evidence.get(cid, [])
        for i in range(5):
            # 如果当前索引有符合条件的证据则填入，否则填空字符串
            content = source_list[i] if i < len(source_list) else ""
            retrieved_evidence[str(i)] = {"content": content}

        new_item["evidence"] = retrieved_evidence
        new_data.append(new_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"[Saved] Retrieved claim–evidence JSON -> {output_path}")


# ===============================
# Step 3: 主流程（仅检索 + 统计 + 保存）
# ===============================

def main():
    data_path = Path("data/health_info.json")
    output_path = Path("data/health_info_retrieved.json")
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    raw_data, claim_texts, claim_labels, evidence_texts = load_health_info(data_path)
    print(f"#Claims: {len(claim_texts)}")
    print(f"#Evidences: {len(evidence_texts)}")
    
    fallback_evidence = {}
    for cid, item in enumerate(raw_data):
        evs = [ev.get("content", "") for ev in item.get("evidence", {}).values() if ev.get("content", "")]
        fallback_evidence[cid] = evs

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

    search_results = pipeline.collection.search(
        data=claim_embeddings.tolist(),
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 16}},
        limit=5,
    )

    # ---- 构建 claim -> evidence 映射 ----
    claim_to_hits = defaultdict(list)
    evidence_to_claims = defaultdict(list)

    for cid, hits in enumerate(search_results):
        for hit in hits:
            # 限制 1：只保留语义相似度高于 0.7 的证据
            if hit.distance > 0.75:
                claim_to_hits[cid].append(hit.id)
                evidence_to_claims[hit.id].append(cid)
                
    # ---- 统计回退比例 ----        
    fallback_count = sum(1 for cid in range(len(claim_texts)) if cid not in claim_to_hits)
    print(f"\n[Threshold Analysis] Claims with NO evidence > 0.75: {fallback_count}/{len(claim_texts)} ({fallback_count/len(claim_texts):.2%})")

    # ---- 统计（不作图，仅打印 / 后续使用） ----
    reuse_stats = reuse_statistics(evidence_to_claims)
    label_stats = label_consistency(evidence_to_claims, claim_labels)

    print("\n[Evidence Reuse Statistics]")
    for k, v in reuse_stats.items():
        print(f"{k}: {v}")

    print("\n[Label Consistency]")
    for k, v in label_stats.items():
        print(f"{k}: {v}")

    # ---- 另存 JSON ----
    save_retrieved_pairs(
        original_data=raw_data,
        evidence_texts=evidence_texts,
        claim_to_hits=claim_to_hits,
        fallback_evidence=fallback_evidence,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()