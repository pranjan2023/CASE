from verification.scoring import p_value, bh_threshold

def update(state):

    print("Update stage")

    mean = state.last_mean
    std  = state.last_std
    n    = state.last_n

    # --- ACCEPTANCE LOGIC (STRICT ORDERING) ---

    if state.best_mean is None:
        accept = True
        p = None
        threshold = None

    else:
        # HARD CONSTRAINT: must improve AND be cross-valid
        if mean <= state.best_mean or not getattr(state, "cross_valid", True):
            accept = False
            p = None
            threshold = None
        else:
            p = p_value(
                {"mean": mean, "std": std},
                {"mean": state.best_mean, "std": state.best_std}
            )

            state.p_values.append(p)
            threshold = bh_threshold(state.p_values)

            if threshold is None:
                accept = False
            else:
                accept = p <= threshold

    # --- UPDATE BEST ---
    if accept:
        state.best_mean = mean
        state.best_std  = std
        state.best_geometry = state.geometry

    # --- HISTORY LOG ---
    state.history.append({
        "geometry": state.geometry,
        "config_id": state.results.get("FTFP_BERT", {}).get("config_id"),
        "mean": mean,
        "std": std,
        "n": n
    })

    # --- LOGGING ---
    print("mean:", mean, "std:", std, "n:", n)

    if p is not None:
        print("p-value:", p)
        print("BH threshold:", threshold)

    print("Cross-valid:", getattr(state, "cross_valid", True))
    print("Best mean so far:", state.best_mean)

    # --- DAG ---
    config_id = state.results.get("FTFP_BERT", {}).get("config_id")

    node_id = state.dag.add_experiment(
        config_id,
        state.geometry,
        {"mean": mean, "std": std, "n": n}
    )

    print("DAG node:", node_id)

    return state