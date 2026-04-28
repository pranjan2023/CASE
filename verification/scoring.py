import numpy as np

def ccr(new, best):
    delta = new["mean"] - best["mean"]
    unc = (new["std"]**2 + best["std"]**2) ** 0.5
    return delta / unc if unc > 0 else 0.0

import math

def p_value(new, best):
    delta = new["mean"] - best["mean"]
    unc = (new["std"]**2 + best["std"]**2) ** 0.5
    if unc == 0:
        return 1.0
    z = delta / unc
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    p = 1 - cdf
    return max(p, 1e-12)

def bh_threshold(pvals, alpha=0.1):
    m = len(pvals)
    sorted_p = sorted(pvals)
    threshold = None
    for i, p in enumerate(sorted_p, start=1):
        if p <= (i / m) * alpha:
            threshold = p
    return threshold

def aggregate_multi(results):
    stats = {}
    for pl, res in results.items():
        scores = res["scores"]
        stats[pl] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "n": len(scores)
        }
    return stats

def cross_physics_consistency(stats):
    means = [v["mean"] for v in stats.values()]
    stds  = [v["std"] for v in stats.values()]
    # simple criterion: overlap
    max_mean = max(means)
    min_mean = min(means)
    pooled_unc = sum(s**2 for s in stds) ** 0.5
    ratio = (max_mean - min_mean) / pooled_unc if pooled_unc > 0 else 0
    return ratio < 1.5
