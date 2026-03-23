import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def normalize_text(text):
    if text is None:
        return ""
    return str(text).strip()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def safe_mean(values):
    if len(values) == 0:
        return None
    return float(np.mean(values))


def safe_max(values):
    if len(values) == 0:
        return None
    return float(np.max(values))


def summarize_array(values):
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "q25": None,
            "q75": None,
            "improved_ratio": None,
            "degraded_ratio": None,
            "unchanged_ratio": None,
        }

    arr = np.asarray(values, dtype=np.float32)
    improved = float((arr > 0).mean())
    degraded = float((arr < 0).mean())
    unchanged = float((arr == 0).mean())

    return {
        "count": int(arr.shape[0]),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "improved_ratio": improved,
        "degraded_ratio": degraded,
        "unchanged_ratio": unchanged,
    }


def analyze_replacement_effect(
    prev_json_path,
    retrieved_json_path,
    claim_embeddings_path,
    prev_embeddings_path,
    retrieved_embeddings_path,
    output_json_path,
):
    print("[1/5] Loading JSON files...")
    prev_data = load_json(prev_json_path)
    retrieved_data = load_json(retrieved_json_path)

    if len(prev_data) != len(retrieved_data):
        raise ValueError(
            f"JSON length mismatch: prev={len(prev_data)}, retrieved={len(retrieved_data)}"
        )

    print("[2/5] Loading embedding caches...")
    claim_embeddings = np.load(claim_embeddings_path)
    prev_embeddings = np.load(prev_embeddings_path)
    retrieved_embeddings = np.load(retrieved_embeddings_path)

    num_claims = len(prev_data)
    if claim_embeddings.shape[0] != num_claims:
        raise ValueError(
            f"Claim embedding mismatch: claims={num_claims}, claim_embeddings={claim_embeddings.shape[0]}"
        )
    if prev_embeddings.shape[:2] != (num_claims, 5):
        raise ValueError(f"Prev embedding shape must be (N,5,D), got {prev_embeddings.shape}")
    if retrieved_embeddings.shape[:2] != (num_claims, 5):
        raise ValueError(
            f"Retrieved embedding shape must be (N,5,D), got {retrieved_embeddings.shape}"
        )

    print("[3/5] Comparing evidence slots and computing similarities...")
    slot_counts = {
        "retained": 0,
        "replaced": 0,
        "added": 0,
        "deleted": 0,
        "both_empty": 0,
    }

    grouped = {
        str(i): {
            "claim_count": 0,
            "comparable_claim_count": 0,
            "prev_mean_sim": [],
            "retrieved_mean_sim": [],
            "prev_max_sim": [],
            "retrieved_max_sim": [],
            "delta_mean_sim": [],
            "delta_max_sim": [],
        }
        for i in range(6)
    }

    overall_prev_mean = []
    overall_retrieved_mean = []
    overall_prev_max = []
    overall_retrieved_max = []
    overall_delta_mean = []
    overall_delta_max = []

    for claim_idx in range(num_claims):
        prev_item = prev_data[claim_idx]
        retrieved_item = retrieved_data[claim_idx]

        prev_evidence = prev_item.get("evidence", {})
        retrieved_evidence = retrieved_item.get("evidence", {})

        prev_valid_mask = np.linalg.norm(prev_embeddings[claim_idx], axis=1) > 0
        retrieved_valid_mask = np.linalg.norm(retrieved_embeddings[claim_idx], axis=1) > 0

        prev_count = int(prev_valid_mask.sum())
        grouped[str(prev_count)]["claim_count"] += 1

        for slot in range(5):
            key = str(slot)
            prev_text = normalize_text(prev_evidence.get(key, {}).get("content", ""))
            retrieved_text = normalize_text(retrieved_evidence.get(key, {}).get("content", ""))

            prev_non_empty = bool(prev_text)
            retrieved_non_empty = bool(retrieved_text)

            if prev_non_empty and retrieved_non_empty:
                if prev_text == retrieved_text:
                    slot_counts["retained"] += 1
                else:
                    slot_counts["replaced"] += 1
            elif (not prev_non_empty) and retrieved_non_empty:
                slot_counts["added"] += 1
            elif prev_non_empty and (not retrieved_non_empty):
                slot_counts["deleted"] += 1
            else:
                slot_counts["both_empty"] += 1

        claim_vec = claim_embeddings[claim_idx]
        prev_sims = np.dot(prev_embeddings[claim_idx][prev_valid_mask], claim_vec)
        retrieved_sims = np.dot(retrieved_embeddings[claim_idx][retrieved_valid_mask], claim_vec)

        prev_mean = safe_mean(prev_sims)
        retrieved_mean = safe_mean(retrieved_sims)
        prev_max = safe_max(prev_sims)
        retrieved_max = safe_max(retrieved_sims)

        comparable = prev_mean is not None and retrieved_mean is not None
        if comparable:
            delta_mean = float(retrieved_mean - prev_mean)
            delta_max = float(retrieved_max - prev_max)

            grouped[str(prev_count)]["comparable_claim_count"] += 1
            grouped[str(prev_count)]["prev_mean_sim"].append(float(prev_mean))
            grouped[str(prev_count)]["retrieved_mean_sim"].append(float(retrieved_mean))
            grouped[str(prev_count)]["prev_max_sim"].append(float(prev_max))
            grouped[str(prev_count)]["retrieved_max_sim"].append(float(retrieved_max))
            grouped[str(prev_count)]["delta_mean_sim"].append(delta_mean)
            grouped[str(prev_count)]["delta_max_sim"].append(delta_max)

            overall_prev_mean.append(float(prev_mean))
            overall_retrieved_mean.append(float(retrieved_mean))
            overall_prev_max.append(float(prev_max))
            overall_retrieved_max.append(float(retrieved_max))
            overall_delta_mean.append(delta_mean)
            overall_delta_max.append(delta_max)

    total_slots = num_claims * 5
    transition_summary = {
        "total_slots": total_slots,
        "retained": {
            "count": slot_counts["retained"],
            "ratio": float(slot_counts["retained"] / total_slots),
        },
        "replaced": {
            "count": slot_counts["replaced"],
            "ratio": float(slot_counts["replaced"] / total_slots),
        },
        "added": {
            "count": slot_counts["added"],
            "ratio": float(slot_counts["added"] / total_slots),
        },
        "deleted": {
            "count": slot_counts["deleted"],
            "ratio": float(slot_counts["deleted"] / total_slots),
        },
        "both_empty": {
            "count": slot_counts["both_empty"],
            "ratio": float(slot_counts["both_empty"] / total_slots),
        },
    }

    group_summary = {}
    for key, value in grouped.items():
        group_summary[key] = {
            "claim_count": int(value["claim_count"]),
            "comparable_claim_count": int(value["comparable_claim_count"]),
            "prev_mean_sim": summarize_array(value["prev_mean_sim"]),
            "retrieved_mean_sim": summarize_array(value["retrieved_mean_sim"]),
            "prev_max_sim": summarize_array(value["prev_max_sim"]),
            "retrieved_max_sim": summarize_array(value["retrieved_max_sim"]),
            "delta_mean_sim": summarize_array(value["delta_mean_sim"]),
            "delta_max_sim": summarize_array(value["delta_max_sim"]),
        }

    result = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "claim_count": int(num_claims),
            "evidence_slots_per_claim": 5,
            "paths": {
                "prev_json": str(prev_json_path),
                "retrieved_json": str(retrieved_json_path),
                "claim_embeddings": str(claim_embeddings_path),
                "prev_embeddings": str(prev_embeddings_path),
                "retrieved_embeddings": str(retrieved_embeddings_path),
            },
        },
        "evidence_transition": transition_summary,
        "similarity_improvement": {
            "overall": {
                "comparable_claim_count": len(overall_delta_mean),
                "prev_mean_sim": summarize_array(overall_prev_mean),
                "retrieved_mean_sim": summarize_array(overall_retrieved_mean),
                "prev_max_sim": summarize_array(overall_prev_max),
                "retrieved_max_sim": summarize_array(overall_retrieved_max),
                "delta_mean_sim": summarize_array(overall_delta_mean),
                "delta_max_sim": summarize_array(overall_delta_max),
            },
            "grouped_by_prev_valid_evidence_count": group_summary,
        },
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("[4/5] Key summary:")
    print(
        "  Transition - retained/replaced/added/deleted:",
        transition_summary["retained"]["count"],
        transition_summary["replaced"]["count"],
        transition_summary["added"]["count"],
        transition_summary["deleted"]["count"],
    )
    print(
        "  Overall comparable claims:",
        result["similarity_improvement"]["overall"]["comparable_claim_count"],
    )
    print(
        "  Overall prev/retrieved mean_sim mean:",
        result["similarity_improvement"]["overall"]["prev_mean_sim"]["mean"],
        result["similarity_improvement"]["overall"]["retrieved_mean_sim"]["mean"],
    )
    print(
        "  Overall prev/retrieved max_sim mean:",
        result["similarity_improvement"]["overall"]["prev_max_sim"]["mean"],
        result["similarity_improvement"]["overall"]["retrieved_max_sim"]["mean"],
    )
    print(
        "  Overall Δmean_sim mean:",
        result["similarity_improvement"]["overall"]["delta_mean_sim"]["mean"],
    )
    print(
        "  Overall Δmax_sim mean:",
        result["similarity_improvement"]["overall"]["delta_max_sim"]["mean"],
    )
    print(f"[5/5] Saved summary JSON -> {output_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze evidence replacement/addition effect with cached embeddings."
    )
    parser.add_argument("--prev-json", default="data/health_info.json")
    parser.add_argument("--retrieved-json", default="data/health_info_retrieved.json")
    parser.add_argument("--claim-embeddings", default="cache/claims_embeddings.npy")
    parser.add_argument("--prev-embeddings", default="cache/evidences_embeddings_prev.npy")
    parser.add_argument("--retrieved-embeddings", default="cache/evidences_embeddings_r.npy")
    parser.add_argument(
        "--output-json",
        default="outputs/feasibility/replacement_effect_summary.json",
    )

    args = parser.parse_args()

    analyze_replacement_effect(
        prev_json_path=Path(args.prev_json),
        retrieved_json_path=Path(args.retrieved_json),
        claim_embeddings_path=Path(args.claim_embeddings),
        prev_embeddings_path=Path(args.prev_embeddings),
        retrieved_embeddings_path=Path(args.retrieved_embeddings),
        output_json_path=Path(args.output_json),
    )


if __name__ == "__main__":
    main()
