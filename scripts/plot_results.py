#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import matplotlib.pyplot as plt
from pathlib import Path

def load_agent_history(mode):
    path = Path(f"test_runs/{mode}")
    if not path.exists():
        return None, None
    checkpoints = sorted(path.glob("checkpoint_*.json"))
    iterations = []
    best_means = []
    for ckpt in checkpoints:
        with open(ckpt) as f:
            data = json.load(f)
            iterations.append(data["iteration"])
            best_means.append(data["best_mean"])
    return iterations, best_means

if __name__ == "__main__":
    plt.figure(figsize=(10,6))
    for mode in ["heuristic", "llm"]:
        iters, means = load_agent_history(mode)
        if iters:
            plt.plot(iters, means, label=mode)
    try:
        with open("random_baseline.json") as f:
            rb = json.load(f)
            plt.axhline(y=rb["best_mean"], color='gray', linestyle='--', label=f'Random best = {rb["best_mean"]:.4f}')
    except:
        pass
    plt.xlabel("Iteration")
    plt.ylabel("Best mean efficiency (sampled)")
    plt.title("Agent Performance Comparison")
    plt.legend()
    plt.grid()
    plt.savefig("agent_performance.png", dpi=150)
    plt.show()