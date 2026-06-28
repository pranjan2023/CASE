#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from phase3.hzz_oracle import run_oracle

def random_scan(n=100, runs=50, events=2000):
    best_mean = -1
    best_geom = None
    for i in range(n):
        geom = {
            "pt_cut": np.random.uniform(5, 50),
            "eta_cut": np.random.uniform(1.5, 3.0),
            "iso_cut": np.random.uniform(0.1, 0.5),
        }
        config = {
            "pt_cut": geom["pt_cut"],
            "eta_cut": geom["eta_cut"],
            "iso_cut": geom["iso_cut"],
            "runs": runs,
            "events": events,
        }
        result = run_oracle(config)
        mean = np.mean(result["scores"])
        if mean > best_mean:
            best_mean = mean
            best_geom = geom
        if (i+1) % 10 == 0:
            print(f"Random scan {i+1}/{n}: best so far = {best_mean:.5f}")
    # Save result
    import json
    with open("random_baseline.json", "w") as f:
        json.dump({"best_geom": best_geom, "best_mean": best_mean}, f)
    return best_geom, best_mean

if __name__ == "__main__":
    best_geom, best_mean = random_scan(100)
    print(f"\nBest random config: {best_geom} -> mean efficiency {best_mean:.5f}")