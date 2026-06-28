#!/usr/bin/env python3
"""
run_phase3.py — Entry point for Phase 3 (HZZ or mock) and Phase 4 (mock geometry).
Runs loop: Plan → Execute → Observe → Verify → Update.
"""

import argparse
import json
from pathlib import Path
from core.state import SystemState
from core.utils import clean_for_json
from modules.plan import plan
from modules.execute import execute
from modules.observe import observe
from modules.verify import verify
from modules.update import update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of iterations to run")
    parser.add_argument("--pileup", action="store_true",
                        help="Enable pile‑up overlay (only for Phase 3 HZZ)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--mock", action="store_true",
                        help="Use Phase 3 mock oracle (pt, eta, iso) instead of HZZ")
    parser.add_argument("--phase4", action="store_true",
                        help="Use Phase 4 mock geometry oracle (8 parameters)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM, use heuristic only")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for run_log.json (default: auto-detected)")
    args = parser.parse_args()

    # ---------- Determine mode and output directory ----------
    if args.phase4:
        mode = "phase4"
        default_out = "test_runs/phase4"
        runs, events = 50, 2000
        use_mock = False
    elif args.mock:
        mode = "phase3"
        default_out = "test_runs/mock"
        runs, events = 50, 2000
        use_mock = True
    else:
        mode = "phase3"
        default_out = "phase3"
        runs, events = 20, 1000
        use_mock = False

    out_dir = Path(args.output) if args.output else Path(default_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load LLM if not disabled ----------
    if not args.no_llm:
        try:
            from modules import mlx_planner
            from mlx_lm import load
            print("Loading LLM model...")
            model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
            mlx_planner.set_model(model, tokenizer)
            print("LLM loaded.")
        except Exception as e:
            print(f"LLM load failed: {e}. Proceeding without LLM.")

    # ---------- Initialise state ----------
    state = SystemState(
        mode=mode,
        iteration=0,
        geometry={},
        best_mean=None,
        history=[],
        rejected=[],
        epistemic_flags=[],
        p_values=[],
        oracle_runs=runs,
        oracle_events=events,
        pileup=args.pileup,
        use_mock_oracle=use_mock,
    )

    print(f"Starting loop: mode={mode}, iterations={args.iterations}")
    print(f"Saving logs to: {out_dir / 'run_log.json'}")

    # ---------- Run iterations ----------
    for i in range(args.iterations):
        state.iteration = i
        state = plan(state)
        state = execute(state)
        state = observe(state)
        state = verify(state)
        state = update(state)

        # Build checkpoint and clean numpy types
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
        clean_checkpoint = clean_for_json(checkpoint)
        with open(out_dir / "run_log.json", "w") as f:
            json.dump(clean_checkpoint, f, indent=2)

        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{args.iterations}: best_mean = {state.best_mean:.4f}")

    print("\n=== Run complete ===")
    print(f"Best mean: {state.best_mean:.4f}")
    print(f"Best geometry: {state.best_geometry}")
    print(f"Total rejections: {len(state.rejected)}")
    print(f"Log saved to {out_dir / 'run_log.json'}")


if __name__ == "__main__":
    main()