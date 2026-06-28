"""
update.py — Phase 3 update stage.

- Accept/reject based on BH-FDR gate and cross-physics consistency.
- Logs rejected configurations into state.rejected with detailed reasons.
- Updates the latest EpistemicFlags with the rejection reason.
"""

from verification.scoring import p_value, bh_threshold
from core.state import RejectedExperiment


def update(state):

    print("Update stage")

    mean = state.last_mean
    std  = state.last_std
    n    = state.last_n
    geom = state.geometry

    # --- Determine acceptance ---

    accept = False
    p = None
    threshold = None
    rejection_reason = None

    if state.best_mean is None:
        accept = True
        rejection_reason = None
    else:
        # 1. Must improve AND be cross-valid
        if mean <= state.best_mean:
            accept = False
            rejection_reason = "no_improvement"
        elif not getattr(state, "cross_valid", True):
            accept = False
            rejection_reason = "cross_physics_divergence"
        else:
            # 2. Compute p-value and check BH threshold
            p = p_value(
                {"mean": mean, "std": std, "jackknife_std": state.last_jackknife_std},
                {"mean": state.best_mean, "std": state.best_std, "jackknife_std": state.best_jackknife_std}
            )
            state.p_values.append(p)
            threshold = bh_threshold(state.p_values)

            if threshold is None:
                accept = False
                rejection_reason = "bh_gate"
            else:
                accept = p <= threshold
                if not accept:
                    rejection_reason = "bh_gate"

    # --- If rejected, log it into state.rejected ---
    if not accept and rejection_reason is not None:
        # Get AD stat from latest epistemic flags if available
        ad_stat = None
        if state.epistemic_flags:
            ad_stat = state.epistemic_flags[-1].anderson_darling_stat

        rejected_exp = RejectedExperiment(
            geometry=geom,
            mean=mean,
            std=std,
            rejection_reason=rejection_reason,
            cross_physics_stat=ad_stat,
            best_mean_at_rejection=state.best_mean,
        )
        state.rejected.append(rejected_exp)

        # Update the latest epistemic flag with the rejection reason
        if state.epistemic_flags:
            state.epistemic_flags[-1].rejection_reason = rejection_reason

    # --- Update best if accepted ---
    if accept:
        state.best_mean = mean
        state.best_std = std
        state.best_geometry = geom
        # Store jackknife std for p-value calculations
        prim = state.multi_stats.get("FTFP_BERT", {})
        state.best_jackknife_std = prim.get("jackknife_std", std)

    # --- Store latest jackknife std for next iteration ---
    prim = state.multi_stats.get("FTFP_BERT", {})
    state.last_jackknife_std = prim.get("jackknife_std", std)

    # --- History log ---
    state.history.append({
        "geometry": geom,
        "config_id": state.results.get("FTFP_BERT", {}).get("config_id"),
        "mean": mean,
        "std": std,
        "n": n,
        "accepted": accept,
        "rejection_reason": rejection_reason,
    })

    # --- Logging ---
    print(f"  mean: {mean:.4f}, std: {std:.4f}, n: {n}")
    if p is not None:
        print(f"  p-value: {p:.6f}, BH threshold: {threshold}")
    print(f"  Cross-valid: {getattr(state, 'cross_valid', True)}")
    print(f"  Accepted: {accept}")
    if not accept and rejection_reason:
        print(f"  Rejection reason: {rejection_reason}")

    # --- DAG logging ---
    config_id = state.results.get("FTFP_BERT", {}).get("config_id")
    if config_id:
        node_id = state.dag.add_experiment(
            config_id,
            geom,
            {"mean": mean, "std": std, "n": n}
        )
        print(f"  DAG node: {node_id}")

    return state