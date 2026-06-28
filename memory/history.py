import json
import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_LOG = os.path.join(BASE_DIR, "memory", "experiment_history.json")
PHASE3_LOG = os.path.join(BASE_DIR, "phase3", "run_log.json")


def load_history():
    """Load all experiment history from legacy and Phase 3 logs."""
    history = []

    # 1. Legacy log
    if os.path.exists(LEGACY_LOG):
        with open(LEGACY_LOG, "r") as f:
            try:
                legacy = json.load(f)
                if isinstance(legacy, list):
                    history.extend(legacy)
            except json.JSONDecodeError:
                pass

    # 2. Phase 3 run log (most recent checkpoint)
    if os.path.exists(PHASE3_LOG):
        with open(PHASE3_LOG, "r") as f:
            try:
                phase3 = json.load(f)
                if isinstance(phase3, dict) and "history" in phase3:
                    history.extend(phase3["history"])
            except json.JSONDecodeError:
                pass

    return history


def save_history(entries):
    """Save history to the legacy log file (overwrites)."""
    os.makedirs(os.path.dirname(LEGACY_LOG), exist_ok=True)
    with open(LEGACY_LOG, "w") as f:
        json.dump(entries, f, indent=2)