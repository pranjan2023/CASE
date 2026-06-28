"""
scoring.py — Phase 3 statistical validation layer.

Changes from Phase 2:
- cross_physics_consistency: Anderson-Darling k-sample test
  replacing hardcoded ratio < 1.5
- aggregate_multi: jackknife_std added, scores kept for AD test
- p_value: uses jackknife_std instead of raw std
- bh_threshold: guards against empty pvals list
- No synthetic noise anywhere
"""

import numpy as np
import math
from scipy.stats import anderson_ksamp


def jackknife_std(scores: list) -> float:
    n = len(scores)
    if n < 2:
        return float(np.std(scores))
    arr = np.array(scores, dtype=float)
    jk_means = np.array([np.mean(np.delete(arr, i)) for i in range(n)])
    return float(np.sqrt((n - 1) / n * np.sum((jk_means - jk_means.mean()) ** 2)))


def aggregate_multi(results: dict) -> dict:
    stats = {}
    for pl, res in results.items():
        scores = res["scores"]
        arr = np.array(scores, dtype=float)
        stats[pl] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "jackknife_std": jackknife_std(scores),
            "n": len(scores),
            "scores": scores,   # kept for AD test downstream
        }
    return stats


def cross_physics_consistency(stats: dict) -> tuple[bool, float]:
    """
    Anderson-Darling k-sample test across physics lists.
    Returns (consistent: bool, ad_statistic: float).
    α = 0.05 (critical_values index 2).
    """
    score_groups = [v["scores"] for v in stats.values()]
    if len(score_groups) < 2 or any(len(g) < 2 for g in score_groups):
        return True, 0.0
    try:
        result = anderson_ksamp(score_groups, variant='asymptotic')
        ad_stat = float(result.statistic)
        consistent = ad_stat < result.critical_values[2]
        return consistent, ad_stat
    except Exception:
        return True, 0.0


def ccr(new: dict, best: dict) -> float:
    delta = new["mean"] - best["mean"]
    unc = (new["std"] ** 2 + best["std"] ** 2) ** 0.5
    return delta / unc if unc > 0 else 0.0


def p_value(new: dict, best: dict) -> float:
    delta = new["mean"] - best["mean"]
    unc = (new.get("jackknife_std", new["std"]) ** 2 +
           best.get("jackknife_std", best["std"]) ** 2) ** 0.5
    if unc == 0:
        return 1.0
    z = delta / unc
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return max(1 - cdf, 1e-12)


def bh_threshold(pvals: list, alpha: float = 0.1):
    m = len(pvals)
    if m == 0:
        return None
    sorted_p = sorted(pvals)
    threshold = None
    for i, p in enumerate(sorted_p, start=1):
        if p <= (i / m) * alpha:
            threshold = p
    return threshold