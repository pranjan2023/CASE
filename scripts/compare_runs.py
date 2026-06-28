#!/usr/bin/env python3
"""
compare_runs.py — Compare convergence speed of multiple runs.

Usage:
    python scripts/compare_runs.py --logs test_runs/run1/run_log.json test_runs/run2/run_log.json --labels "Heuristic" "LLM" --optimum 0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_best_means(log_path):
    with open(log_path) as f:
        data = json.load(f)
    history = data.get("history", [])
    best_means = []
    best_so_far = -np.inf
    for h in history:
        mean = h.get("mean", -np.inf)
        if mean > best_so_far:
            best_so_far = mean
        best_means.append(best_so_far)
    return np.array(best_means)


def compute_metrics(best_means, final_best):
    target = 0.9 * final_best
    idx = np.argmax(best_means >= target)
    if idx == 0 and best_means[0] >= target:
        iter_90 = 0
    elif idx == 0:
        iter_90 = len(best_means)
    else:
        iter_90 = idx
    try:
        auc = np.trapezoid(best_means, dx=1.0)
    except AttributeError:
        auc = np.trapz(best_means, dx=1.0)
    return iter_90, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True,
                        help="Paths to run_log.json files")
    parser.add_argument("--labels", nargs="+",
                        help="Labels for each run (must match number of logs)")
    parser.add_argument("--optimum", type=float, default=None,
                        help="Known optimum value (for reference)")
    parser.add_argument("--save", default="plots/convergence_comparison.png",
                        help="Output plot filename (default: plots/convergence_comparison.png)")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.logs):
        print("Number of labels must match number of logs.")
        return
    if args.labels is None:
        args.labels = [f"Run {i+1}" for i in range(len(args.logs))]

    # Ensure output directory exists
    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    metrics = []
    for log_path, label in zip(args.logs, args.labels):
        best_means = load_best_means(log_path)
        iterations = np.arange(len(best_means))
        final_best = best_means[-1]
        iter_90, auc = compute_metrics(best_means, final_best)
        metrics.append((label, final_best, iter_90, auc))
        plt.plot(iterations, best_means, label=f"{label} (final={final_best:.3f})", linewidth=2)

        target = 0.9 * final_best
        idx = np.argmax(best_means >= target)
        if idx < len(best_means):
            plt.plot(idx, best_means[idx], 'o', markersize=8, label=f"{label} 90% at iter {idx}")

    if args.optimum is not None:
        plt.axhline(y=args.optimum, color='black', linestyle='--', label=f"Optimum = {args.optimum}")

    plt.xlabel("Iteration")
    plt.ylabel("Best mean")
    plt.title("Convergence comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.save, dpi=150)
    print(f"Plot saved to {args.save}")

    print("\n--- Convergence Metrics ---")
    print(f"{'Run':<15} {'Final best':<12} {'Iter to 90%':<12} {'AUC':<12}")
    for label, final, iter90, auc in metrics:
        print(f"{label:<15} {final:<12.4f} {iter90:<12} {auc:<12.2f}")

    if len(metrics) >= 2:
        base_iter = metrics[0][2]
        print("\n--- Relative to first run ---")
        for label, _, iter90, _ in metrics[1:]:
            speedup = base_iter / iter90 if iter90 > 0 else np.inf
            print(f"{label}: {speedup:.2f}x speedup (based on 90% iteration)")


if __name__ == "__main__":
    main()