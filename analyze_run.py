#!/usr/bin/env python3
"""
analyze_run.py - Load a test run and generate plots for each parameter,
acceptance status, and best mean over iterations.

Usage:
    python analyze_run.py --run_dir test_runs/heuristic/
    python analyze_run.py --run_dir test_runs/llm/ --save_csv
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_checkpoints(run_dir):
    """Load all checkpoint JSON files from run_dir."""
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory {run_path} not found.")
    checkpoints = sorted(run_path.glob("checkpoint_*.json"))
    if not checkpoints:
        raise ValueError(f"No checkpoint files found in {run_path}")
    data = []
    for ckpt in checkpoints:
        with open(ckpt) as f:
            d = json.load(f)
            data.append(d)
    return data

def extract_history(data):
    """Extract per-iteration info: iteration, geom, best_mean, acceptance, reason."""
    records = []
    for d in data:
        it = d["iteration"]
        geom = d["geometry"]
        best_mean = d["best_mean"]
        # Find history entry for this iteration
        history = d.get("history", [])
        # The history may contain multiple entries, but we need the one for this iteration.
        # We'll match by iteration number stored in history entry.
        # However, history entries don't store iteration number; we assume the history list is ordered.
        # The last entry in history corresponds to the current iteration, but we need to ensure.
        # We can simply look for the entry with matching config_id? Easier: we can get acceptance from the update stage if stored.
        # In our checkpoints, we don't store acceptance per iteration explicitly.
        # We can deduce: if best_mean changed, it was accepted; otherwise rejected.
        # But best_mean can also stay same if rejected. Also, we have 'rejected' list with geometries.
        # We'll mark acceptance based on whether the current geometry is in the best_geometry of that iteration.
        # Actually we can compute: if geometry equals best_geometry, then accepted, else rejected.
        # But careful: the best may have been set in a previous iteration.
        # We'll compute: if d["best_geometry"] == geom: accepted.
        # But best_geometry is the global best at that iteration.
        best_geom = d["best_geometry"]
        accepted = (best_geom == geom)
        # Get rejection reason if any
        reason = None
        # We can check if this geometry is in the rejected list
        rejected_list = d.get("rejected", [])
        for r in rejected_list:
            if r["geometry"] == geom:
                reason = r["rejection_reason"]
                break
        # If not in rejected and not accepted, it's a "no_improvement" but not logged? 
        # Actually if not accepted and not rejected, it means it was a "no_improvement" (we didn't log it).
        # But we can check history entry for rejection reason if stored.
        # In our current update.py, we log rejected only when reject=True. For no_improvement, we also log rejected.
        # So if not accepted, it should be in rejected list.
        # If not found, reason = "no_improvement" (default).
        if not accepted and reason is None:
            reason = "no_improvement"
        records.append({
            "iteration": it,
            "pt_cut": geom.get("pt_cut"),
            "eta_cut": geom.get("eta_cut"),
            "iso_cut": geom.get("iso_cut"),
            "best_mean": best_mean,
            "accepted": accepted,
            "rejection_reason": reason,
        })
    return pd.DataFrame(records)

def plot_analysis(df, run_name, save_csv=False):
    """Generate plots: best mean, parameters, acceptance."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Best mean vs iteration
    ax = axes[0]
    ax.plot(df["iteration"], df["best_mean"], marker='o', linestyle='-', color='blue')
    ax.set_ylabel("Best Mean Efficiency")
    ax.grid(True)
    ax.set_title(f"Run: {run_name}")
    
    # 2. Parameters vs iteration, color by acceptance
    ax = axes[1]
    # Color: green for accepted, red for rejected
    colors = df["accepted"].map({True: 'green', False: 'red'})
    ax.scatter(df["iteration"], df["pt_cut"], c=colors, alpha=0.7, label='pt_cut')
    ax.scatter(df["iteration"], df["eta_cut"], c=colors, marker='s', alpha=0.7, label='eta_cut')
    ax.scatter(df["iteration"], df["iso_cut"], c=colors, marker='^', alpha=0.7, label='iso_cut')
    ax.set_ylabel("Parameter value")
    ax.legend()
    ax.grid(True)
    ax.set_title("Proposed Parameters (color: accepted=green, rejected=red)")
    
    # 3. Rejection reasons distribution (or acceptance rate)
    ax = axes[2]
    # Plot acceptance as binary: 1=accepted, 0=rejected
    ax.bar(df["iteration"], df["accepted"].astype(int), color='green', label='Accepted')
    ax.set_ylabel("Accepted (1) / Rejected (0)")
    ax.set_xlabel("Iteration")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True)
    ax.set_title("Acceptance Status per Iteration")
    
    plt.tight_layout()
    plt.savefig(f"analysis_{run_name}.png", dpi=150)
    print(f"Saved plot to analysis_{run_name}.png")
    
    # Optionally save CSV
    if save_csv:
        csv_path = f"analysis_{run_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved data to {csv_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Path to run directory (e.g., test_runs/heuristic/)")
    parser.add_argument("--save_csv", action="store_true", help="Save data to CSV")
    args = parser.parse_args()
    
    run_name = Path(args.run_dir).name
    data = load_checkpoints(args.run_dir)
    df = extract_history(data)
    plot_analysis(df, run_name, args.save_csv)

if __name__ == "__main__":
    main()