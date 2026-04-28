import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "memory", "experiment_history.json")

def log_experiment(state):

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

        "timestamp": str(datetime.utcnow())
    }

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append(experiment)

    with open(LOG_FILE, "w") as f:
        json.dump(history, f, indent=2)