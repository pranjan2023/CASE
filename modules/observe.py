"""
observe.py — print detailed results of the latest experiment.

Works for both Phase 2 (Geant4) and Phase 3 (HZZ oracle).
"""

def observe(state):
    print(f"\n--- Observe stage (iteration {state.iteration}) ---")

    if not state.history:
        print("  No history yet.")
        return state

    last = state.history[-1]
    geom = last.get("geometry", {})
    mean = last.get("mean", 0.0)
    std = last.get("std", 0.0)
    n = last.get("n", 0)
    accepted = last.get("accepted", False)
    rejection_reason = last.get("rejection_reason", None)

    # Print geometry nicely
    if state.mode == "phase3":
        pt = geom.get("pt_cut", "?")
        eta = geom.get("eta_cut", "?")
        iso = geom.get("iso_cut", "?")
        print(f"  Cuts: pt_cut={pt:.2f} GeV, |eta|<{eta:.3f}, iso<{iso:.3f}")
    else:
        size = geom.get("detector_size_cm", "?")
        mat = geom.get("material", "?")
        print(f"  Geometry: {size:.2f} cm of {mat}")

    # Print results
    print(f"  Mean efficiency: {mean:.5f}  ± {std:.5f}  (n={n})")
    print(f"  Cross‑physics consistent: {state.cross_valid}")

    # Acceptance status
    if accepted:
        print("  ✅ ACCEPTED")
    else:
        print(f"  ❌ REJECTED  (reason: {rejection_reason})")

    # Show best so far
    if state.best_mean is not None:
        print(f"  Best so far: {state.best_mean:.5f}  (iteration {state.iteration - len(state.rejected)})")

    # If there are epistemic flags, show the latest rejection reason (if any)
    if state.epistemic_flags:
        latest_flag = state.epistemic_flags[-1]
        if latest_flag.rejection_reason:
            print(f"  Epistemic: last rejection reason = {latest_flag.rejection_reason}")

    return state