import json
import os
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "memory", "experiment_history.json")


def log_experiment(state):
    # Determine current experiment acceptance status from history
    accepted = None
    rejection_reason = None
    if state.history:
        last = state.history[-1]
        accepted = last.get("accepted")
        rejection_reason = last.get("rejection_reason")

    config_id = state.results.get("FTFP_BERT", {}).get("config_id")

    experiment = {
        "iteration": state.iteration,
        "geometry": state.geometry,
        "config_id": config_id,

        "mean": state.last_mean,
        "std": state.last_std,
        "n": state.last_n,

        "best_mean": state.best_mean,
        "best_geometry": state.best_geometry,

        "cross_valid": getattr(state, "cross_valid", True),
        "multi_stats": getattr(state, "multi_stats", {}),

        "accepted": accepted,
        "rejection_reason": rejection_reason,

        # New fields (safe if missing)
        "rejected_count": len(getattr(state, "rejected", [])),
        "epistemic_flags": [
            f.model_dump() for f in getattr(state, "epistemic_flags", [])
        ] if hasattr(state, "epistemic_flags") and state.epistemic_flags else [],

        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Load existing history
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []

    history.append(experiment)

    # Write back
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(history, f, indent=2)