#!/usr/bin/env python3
"""
run_test.py - Run agent test from project root.
Usage: python run_test.py --mode heuristic --iterations 100
"""

import argparse
import json
from pathlib import Path
from core.state import SystemState
from modules.plan import plan
from modules.execute import execute
from modules.observe import observe
from modules.verify import verify
from modules.update import update


def run_agent(mode, iterations, runs=50, events=2000, pileup=False):
    state = SystemState(
        mode="phase3",
        iteration=0,
        geometry={},
        best_mean=None,
        history=[],
        rejected=[],
        epistemic_flags=[],
        p_values=[],
        oracle_runs=runs,
        oracle_events=events,
        pileup=pileup,
    )

    if mode == "llm":
        try:
            from modules import mlx_planner
            from mlx_lm import load
            model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
            mlx_planner.set_model(model, tokenizer)
            print("LLM model loaded.")
        except Exception as e:
            print(f"Failed to load LLM: {e}. Falling back to heuristic.")
            mode = "heuristic"

    out_dir = Path("test_runs") / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(iterations):
        state.iteration = i
        state = plan(state)
        state = execute(state)
        state = observe(state)
        state = verify(state)
        state = update(state)

        checkpoint = {
            "iteration": i,
            "geometry": state.geometry,
            "best_mean": state.best_mean,
            "best_geometry": state.best_geometry,
            "history": state.history,
            "rejected": [r.model_dump() for r in state.rejected],
            "epistemic_flags": [f.model_dump() for f in state.epistemic_flags],
            "p_values": state.p_values,
            "cross_valid": state.cross_valid,
        }
        with open(out_dir / f"checkpoint_{i:03d}.json", "w") as f:
            json.dump(checkpoint, f, indent=2)

        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{iterations}: best_mean = {state.best_mean:.5f}")

    summary = {
        "mode": mode,
        "iterations": iterations,
        "best_mean": state.best_mean,
        "best_geometry": state.best_geometry,
        "total_rejections": len(state.rejected),
        "rejection_reasons": [r.rejection_reason for r in state.rejected],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Run complete ===")
    print(f"Best mean: {state.best_mean:.5f}")
    print(f"Best geometry: {state.best_geometry}")
    print(f"Total rejections: {len(state.rejected)}")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    run_agent(args.mode, args.iterations)