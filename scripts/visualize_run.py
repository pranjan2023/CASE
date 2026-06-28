#!/usr/bin/env python3
"""
visualize_run.py — Load a run log and generate plots.

Usage:
    python scripts/visualize_run.py --log test_runs/llm_phase4/run_log.json --optimum target_radius=25 target_length=50 absorber_thickness=2 absorber_material=Pb gap_thickness=0.5 n_layers=15 field_strength=1 readout_segmentation=16

The plot is saved to plots/<run_name>_run_plots.png.
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_log(log_path):
    with open(log_path) as f:
        return json.load(f)


def extract_history(data):
    history = data["history"]
    accepted = [h.get("accepted", False) for h in history]
    means = [h["mean"] for h in history]
    geoms = [h["geometry"] for h in history]
    iterations = list(range(len(history)))
    best_means = []
    best_so_far = -np.inf
    for h in history:
        if h["mean"] > best_so_far:
            best_so_far = h["mean"]
        best_means.append(best_so_far)
    return iterations, means, best_means, geoms, accepted


def plot_run(log_path, optimum=None, save_dir="plots"):
    data = load_log(log_path)
    mode = data.get("mode", "unknown")
    iters, means, best_means, geoms, accepted = extract_history(data)

    if not geoms:
        print("No history found.")
        return

    # Determine parameter keys from first geometry
    keys = list(geoms[0].keys())
    # Remove known non‑parameter keys
    exclude = {"config_id", "detector_size_cm", "material", "depth_cm"}
    keys = [k for k in keys if k not in exclude]

    n_params = len(keys)
    if n_params == 0:
        print("No parameter keys found.")
        return

    # Create figure
    fig, axes = plt.subplots(n_params + 1, 1, figsize=(10, 2.5 * (n_params + 1)), sharex=True)

    # Best mean plot
    ax = axes[0]
    ax.plot(iters, best_means, label="Best so far", color='blue', linewidth=2)
    ax.scatter(iters, means, c=['green' if a else 'red' for a in accepted], alpha=0.6, s=30, label='Proposed')
    ax.set_ylabel("Mean score")
    ax.legend()
    ax.grid(True)
    ax.set_title(f"Run: {mode}")

    # Parameter plots
    for i, key in enumerate(keys):
        ax = axes[i + 1]
        vals = [g.get(key, np.nan) for g in geoms]
        ax.scatter(iters, vals, c=['green' if a else 'red' for a in accepted], alpha=0.6, s=30)
        # Mark best accepted value
        best_geom = data.get("best_geometry", {})
        if key in best_geom:
            ax.axhline(y=best_geom[key], color='blue', linestyle='--', label=f"Best {key}")
        # Mark known optimum
        if optimum is not None and key in optimum:
            ax.axhline(y=optimum[key], color='green', linestyle=':', label=f"Optimum {key}")
        ax.set_ylabel(key)
        ax.grid(True)
        if i == n_params - 1:
            ax.set_xlabel("Iteration")
        ax.legend()

    plt.tight_layout()

    # Determine run name from log path parent directory
    run_name = Path(log_path).parent.name
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{run_name}_run_plots.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plots saved to {out_path}")
    plt.show()


def parse_optimum(opt_list):
    """Parse --optimum as key=value pairs, e.g., target_radius=25 target_length=50 ..."""
    opt = {}
    if not opt_list:
        return None
    for pair in opt_list:
        if '=' not in pair:
            continue
        k, v = pair.split('=', 1)
        try:
            opt[k] = float(v)
        except ValueError:
            opt[k] = v  # string values (e.g., material)
    return opt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to run_log.json")
    parser.add_argument("--optimum", nargs='*', help="Known optimum as key=value pairs")
    parser.add_argument("--save_dir", default="plots", help="Directory to save plots")
    args = parser.parse_args()
    opt_dict = parse_optimum(args.optimum) if args.optimum else None
    plot_run(args.log, opt_dict, args.save_dir)