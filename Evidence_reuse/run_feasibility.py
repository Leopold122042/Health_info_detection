import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import kruskal, spearmanr
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
# Step 2: 构建可视化数据
# ===============================

def build_full_evidence_reuse_data(evidence_to_claims, total_evidence_num):
    reuse_counts = np.zeros(total_evidence_num, dtype=int)
    for eid, claims in evidence_to_claims.items():
        reuse_counts[eid] = len(claims)

    zero_cnt = int((reuse_counts == 0).sum())
    print(f"[Full Evidence] reuse=0 evidences: {zero_cnt} ({zero_cnt / total_evidence_num:.2%})")

    return {
        "reuse_counts": reuse_counts.tolist(),
        "zero_reuse_count": zero_cnt,
        "zero_reuse_ratio": float(zero_cnt / total_evidence_num),
    }


def build_similarity_by_reuse_group_data(evidence_to_similarities):
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

    filtered_groups = {k: iqr_filter(v) for k, v in groups.items()}

    return {
        "similarity_groups_raw": {k: [float(x) for x in v] for k, v in groups.items()},
        "similarity_groups_iqr_filtered": {k: [float(x) for x in v] for k, v in filtered_groups.items()},
    }


def hypothesis_test_similarity_trend(evidence_to_similarities):
    reuse_counts = []
    avg_similarities = []

    for sims in evidence_to_similarities.values():
        if sims:
            reuse_counts.append(len(sims))
            avg_similarities.append(float(np.mean(sims)))

    if len(avg_similarities) < 3 or len(set(reuse_counts)) < 2:
        return {
            "valid": False,
            "reason": "有效样本不足，无法进行趋势检验",
        }

    reuse_counts = np.array(reuse_counts, dtype=float)
    avg_similarities = np.array(avg_similarities, dtype=float)

    try:
        rho, p_spearman = spearmanr(reuse_counts, avg_similarities, alternative="less")
    except TypeError:
        rho, p_two_sided = spearmanr(reuse_counts, avg_similarities)
        p_spearman = p_two_sided / 2 if rho < 0 else 1 - p_two_sided / 2

    raw_groups = build_similarity_by_reuse_group_data(evidence_to_similarities)["similarity_groups_raw"]
    ordered_keys = ["0-3", "3-10", "10-20", ">=20"]
    kruskal_samples = [raw_groups.get(k, []) for k in ordered_keys if len(raw_groups.get(k, [])) > 0]

    if len(kruskal_samples) >= 2:
        h_stat, p_kruskal = kruskal(*kruskal_samples)
        kruskal_result = {
            "h_statistic": float(h_stat),
            "p_value": float(p_kruskal),
            "group_count": int(len(kruskal_samples)),
        }
    else:
        kruskal_result = {
            "h_statistic": None,
            "p_value": None,
            "group_count": int(len(kruskal_samples)),
            "note": "非空组数不足2，无法进行Kruskal-Wallis检验",
        }

    alpha = 0.05
    supports_negative_trend = bool((rho < 0) and (p_spearman < alpha))

    return {
        "valid": True,
        "hypothesis": {
            "null": "证据复用次数与平均相似度不存在单调负相关（rho >= 0）",
            "alternative": "证据复用次数与平均相似度存在单调负相关（rho < 0）",
            "alpha": alpha,
        },
        "spearman_negative_trend": {
            "rho": float(rho),
            "p_value": float(p_spearman),
            "sample_size": int(len(avg_similarities)),
            "supports_negative_trend": supports_negative_trend,
        },
        "kruskal_group_difference": kruskal_result,
        "conclusion": "支持“复用次数越大，平均相似度越低”" if supports_negative_trend else "未达到统计显著，无法支持“复用次数越大，平均相似度越低”",
    }


# --- 2.3 Claim 视角 ---

def build_claim_evidence_utilization_data(claim_to_hits, claim_to_origin_ev_cnt, raw_data):
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

    return {
        "origin_evidence_hit_distribution": {str(k): v for k, v in debug_table.items()},
        "origin_evidence_hit_ratio": {
            str(k): [float(x) for x in v.tolist()]
            for k, v in ratios.items()
        },
    }


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
        mask_path = cache_dir / "evd_mask_prev.npy"
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

    full_reuse_plot_data = build_full_evidence_reuse_data(evidence_to_claims, evidence_embeddings.shape[0])
    sim_group_plot_data = build_similarity_by_reuse_group_data(evidence_to_similarities)
    similarity_trend_test = hypothesis_test_similarity_trend(evidence_to_similarities)

    claim_to_hits = {cid: [] for cid in range(len(claim_texts))}
    for eid, cids in evidence_to_claims.items():
        for cid in cids:
            claim_to_hits[cid].append(eid)

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    claim_to_origin_ev_cnt = {i: len(item.get("evidence", {})) for i, item in enumerate(raw_data)}

    claim_utilization_data = build_claim_evidence_utilization_data(claim_to_hits, claim_to_origin_ev_cnt, raw_data)

    output_dir = Path("outputs/feasibility")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "feasibility_metrics.json"

    export_payload = {
        "meta": {
            "claim_count": len(claim_texts),
            "evidence_count": int(evidence_embeddings.shape[0]),
            "search_topk": 5,
            "metric_type": "IP",
            "nprobe": 16,
        },
        "metrics": {
            "reuse_statistics": reuse_stats,
            "label_consistency": label_stats,
            "hypothesis_tests": {
                "similarity_trend_by_reuse": similarity_trend_test,
            },
        },
        "visualization_data": {
            "full_evidence_reuse": full_reuse_plot_data,
            "similarity_by_reuse_group": sim_group_plot_data,
            "claim_evidence_utilization": claim_utilization_data,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, ensure_ascii=False, indent=2)

    print("\n[Hypothesis Test: Similarity Trend by Reuse]")
    if not similarity_trend_test.get("valid", False):
        print(f"Result: {similarity_trend_test.get('reason', '检验失败')}")
    else:
        sp = similarity_trend_test["spearman_negative_trend"]
        print(f"Spearman (one-sided, H1: rho<0): rho={sp['rho']:.4f}, p={sp['p_value']:.6g}, n={sp['sample_size']}")

        kw = similarity_trend_test["kruskal_group_difference"]
        if kw.get("h_statistic") is not None:
            print(f"Kruskal-Wallis: H={kw['h_statistic']:.4f}, p={kw['p_value']:.6g}, groups={kw['group_count']}")
        else:
            print(f"Kruskal-Wallis: {kw.get('note', '无法计算')}")

        print(f"Conclusion: {similarity_trend_test['conclusion']}")

    print(f"[Export] Feasibility metrics saved to: {output_path}")


if __name__ == "__main__":
    main()