# reuse_analysis.py
import numpy as np
from collections import Counter

def reuse_statistics(evidence_to_claims):
    reuse_counts = np.array([
        len(claims)
        for claims in evidence_to_claims.values()
    ])

    return {
        "mean_claims_per_evidence": float(reuse_counts.mean()),
        "max_claims_per_evidence": int(reuse_counts.max()),
        "ratio_reused_>=2": float((reuse_counts >= 2).mean()),
        "ratio_reused_>=5": float((reuse_counts >= 5).mean())
    }

def label_consistency(evidence_to_claims, claim_labels):
    consistency_scores = []

    for claim_ids in evidence_to_claims.values():
        labels = [claim_labels[c] for c in claim_ids]
        majority_ratio = Counter(labels).most_common(1)[0][1] / len(labels)
        consistency_scores.append(majority_ratio)

    return {
        "avg_label_consistency": float(np.mean(consistency_scores))
    }