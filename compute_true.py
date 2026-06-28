#!/usr/bin/env python3
import json
from eval_true_efficiency import true_efficiency

# Agent's best from the run
best = {"pt_cut": 6.68, "eta_cut": 2.795, "iso_cut": 0.5}
best_true = true_efficiency(best["pt_cut"], best["eta_cut"], best["iso_cut"])
print(f"Agent best true efficiency: {best_true:.5f}")

# Loose optimum
loose = {"pt_cut": 5.0, "eta_cut": 3.0, "iso_cut": 0.5}
loose_true = true_efficiency(loose["pt_cut"], loose["eta_cut"], loose["iso_cut"])
print(f"Loose optimum true efficiency: {loose_true:.5f}")

# Random baseline (if you have random_baseline.json)
try:
    with open("random_baseline.json") as f:
        rb = json.load(f)
        # Compute true efficiency of random best geometry
        rb_geom = rb["best_geom"]
        rb_true = true_efficiency(rb_geom["pt_cut"], rb_geom["eta_cut"], rb_geom["iso_cut"])
        print(f"Random baseline true efficiency: {rb_true:.5f}")
except:
    print("Random baseline not found; run random_baseline.py first.")