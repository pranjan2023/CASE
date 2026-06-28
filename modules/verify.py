"""
verify.py — Phase 3 verification layer.

Emits EpistemicFlags per run, using Anderson-Darling for cross‑physics
consistency and jackknife standard errors for real variance.

Rejection reasons are recorded in state.rejected via update.py later.
"""

from verification.scoring import aggregate_multi, cross_physics_consistency
from core.state import EpistemicFlags
import numpy as np


def verify(state):
    print("Verify stage")

    # 1. Aggregate per‑physics‑list stats (includes jackknife_std and raw scores)
    multi_stats = aggregate_multi(state.results)
    state.multi_stats = multi_stats

    # 2. Anderson‑Darling cross‑physics consistency test
    consistent, ad_stat = cross_physics_consistency(multi_stats)
    state.cross_valid = consistent

    # 3. Primary physics list for scalar optimization (keep Phase 2 default)
    primary = "FTFP_BERT"
    prim = multi_stats[primary]

    # 4. Store scalar values (raw std for backward compatibility)
    state.last_mean = prim["mean"]
    state.last_std  = prim["std"]
    state.last_n    = prim["n"]

    # 5. Build and append EpistemicFlags for this iteration
    flags = EpistemicFlags(
        kernel_subordination=True,
        grammar_bounded_search=True,
        search_space_contains_optimum="UNKNOWN",
        cross_physics_consistent=consistent,
        synthetic_noise_injected=False,          # Phase 3: no synthetic noise
        anderson_darling_stat=ad_stat,
        jackknife_std=prim.get("jackknife_std", prim["std"]),
        rejection_reason=None,                   # will be set later in update
    )
    state.epistemic_flags.append(flags)

    print(f"  Cross‑physics valid: {consistent}  (AD stat = {ad_stat:.4f})")
    print(f"  Primary ({primary}) mean = {prim['mean']:.4f} ± {prim['std']:.4f}")

    return state