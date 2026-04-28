from verification.scoring import aggregate_multi, cross_physics_consistency

def verify(state):

    print("Verify stage")

    multi_stats = aggregate_multi(state.results)

    state.multi_stats = multi_stats

    consistent = cross_physics_consistency(multi_stats)
    state.cross_valid = consistent

    # choose primary physics list for optimization
    primary = "FTFP_BERT"

    state.last_mean = multi_stats[primary]["mean"]
    state.last_std  = multi_stats[primary]["std"]
    state.last_n    = multi_stats[primary]["n"]

    print("Cross-physics valid:", consistent)
    print("Stats:", multi_stats)

    return state