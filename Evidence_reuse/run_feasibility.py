import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
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
    return claim_texts, claim_labels, evidence_texts


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
# Step 2: 可视化
# ===============================

def plot_full_evidence_reuse_distribution(evidence_to_claims, total_evidence_num):
    reuse_counts = np.zeros(total_evidence_num, dtype=int)
    for eid, claims in evidence_to_claims.items():
        reuse_counts[eid] = len(claims)

    plt.figure()
    plt.hist(reuse_counts, bins=30, log=True)
    plt.xlabel("#Claims per Evidence")
    plt.ylabel("Evidence Count (log)")
    plt.title("Distribution of Evidence Reuse (Full Evidence View)")
    plt.tight_layout()
    plt.show()

    zero_cnt = int((reuse_counts == 0).sum())
    print(f"[Full Evidence] reuse=0 evidences: {zero_cnt} ({zero_cnt / total_evidence_num:.2%})")


def plot_similarity_by_reuse_group(evidence_to_similarities):
    reuse_to_avg_sim = defaultdict(list)
    for eid, sims in evidence_to_similarities.items():
        if sims:
            reuse_to_avg_sim[len(sims)].append(float(np.mean(sims)))

    groups = {"0-3": [], "3-10": [], "10-20": [], ">=20": []}
    for reuse, sims in reuse_to_avg_sim.items():
        if reuse < 3:
            groups["0-3"].extend(sims)
        elif reuse < 10:
            groups["3-10"].extend(sims)
        elif reuse < 20:
            groups["10-20"].extend(sims)
        else:
            groups[">=20"].extend(sims)

    def iqr_filter(arr):
        if len(arr) < 5:
            return arr
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return [x for x in arr if low <= x <= high]

    data = [iqr_filter(groups[k]) for k in groups]

    plt.figure()
    plt.boxplot(data, tick_labels=list(groups.keys()), showfliers=False)
    plt.xlabel("#Claims per Evidence")
    plt.ylabel("Average Similarity")
    plt.title("Similarity Distribution by Evidence Reuse (Active Evidence)")
    plt.tight_layout()
    plt.show()


# --- 2.3 Claim 视角 ---

def plot_claim_evidence_utilization(claim_to_hits, claim_to_origin_ev_cnt, raw_data):
    # Step 1: Claim -> 原 evidence id 集合（按读取顺序全局编号）
    claim_to_origin_eids = {}
    current_eid = 0
    for cid, item in enumerate(raw_data):
        cnt = len(item.get("evidence", {}))
        claim_to_origin_eids[cid] = set(range(current_eid, current_eid + cnt))
        current_eid += cnt

    # Step 2: 按原始 evidence 数分组
    groups = defaultdict(list)
    for cid, orig_cnt in claim_to_origin_ev_cnt.items():
        groups[orig_cnt].append(cid)

    # Step 3: 统计命中分布
    ratios = {}           # dict[int] -> np.ndarray
    debug_table = {}

    for orig_cnt, cids in sorted(groups.items()):
        hit_dist = np.zeros(orig_cnt + 1, dtype=int)
        for cid in cids:
            hits = claim_to_hits.get(cid, [])
            origin_set = claim_to_origin_eids.get(cid, set())
            hit_origin = len(set(hits) & origin_set)
            hit_dist[hit_origin] += 1
        ratios[orig_cnt] = hit_dist / hit_dist.sum() if hit_dist.sum() > 0 else hit_dist
        debug_table[orig_cnt] = hit_dist.tolist()


# ===============================
# Step 3: 主流程
# ===============================

def main():
    data_path = Path("data/health_info.json")
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    claim_texts, claim_labels, evidence_texts = load_health_info(data_path)
    print(f"#Claims: {len(claim_texts)}")
    print(f"#Evidences: {len(evidence_texts)}")

    encoder = SentenceTransformer("bert-base-chinese")

    claim_embeddings = encode_with_cache(claim_texts, encoder, cache_dir / "claims_embeddings.npy")
    evidence_embeddings = encode_with_cache(evidence_texts, encoder, cache_dir / "evidences_embeddings_prev.npy")
    
    if evidence_embeddings.ndim == 3:
        print(f"[Adjust] Detected 3D evidence embeddings {evidence_embeddings.shape}, flattening...")
        mask_path = cache_dir / "evd_mask.npy"
        if mask_path.exists():
            evd_mask = np.load(mask_path)
            # 根据 mask 筛选有效证据 (mask==1)
            valid_indices = np.where(evd_mask == 1)
            evidence_embeddings = evidence_embeddings[valid_indices]
            print(f"[Adjust] Flattened to {evidence_embeddings.shape} using mask")
        else:
            # 无 mask 则直接展平 (可能包含全 0 向量)
            evidence_embeddings = evidence_embeddings.reshape(-1, evidence_embeddings.shape[-1])
            print(f"[Adjust] Flattened to {evidence_embeddings.shape} without mask")
    
    
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

    evidence_to_claims = defaultdict(list)
    evidence_to_similarities = defaultdict(list)
    for claim_id, hits in enumerate(search_results):
        for hit in hits:
            evidence_to_claims[hit.id].append(claim_id)
            evidence_to_similarities[hit.id].append(hit.score)

    reuse_stats = reuse_statistics(evidence_to_claims)
    label_stats = label_consistency(evidence_to_claims, claim_labels)

    print("\n[Evidence Reuse Statistics]")
    for k, v in reuse_stats.items():
        print(f"{k}: {v}")

    print("\n[Label Consistency]")
    for k, v in label_stats.items():
        print(f"{k}: {v}")

    plot_full_evidence_reuse_distribution(evidence_to_claims, evidence_embeddings.shape[0])
    plot_similarity_by_reuse_group(evidence_to_similarities)

    claim_to_hits = {cid: [] for cid in range(len(claim_texts))}
    for eid, cids in evidence_to_claims.items():
        for cid in cids:
            claim_to_hits[cid].append(eid)

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    claim_to_origin_ev_cnt = {i: len(item.get("evidence", {})) for i, item in enumerate(raw_data)}

    plot_claim_evidence_utilization(claim_to_hits, claim_to_origin_ev_cnt, raw_data)


if __name__ == "__main__":
    main()