"""
plan.py — LLM‑first geometry proposer.

- Calls the MLX planner on every iteration (no random grid pre‑exhaustion).
- Passes state.history, state.best_mean, state.rejected (rejection context),
  and state.epistemic_flags to the planner.
- Validates and clamps the proposed geometry to the search space.
- Uses a simple heuristic fallback if the LLM fails or proposes an invalid config.
"""

import random
import numpy as np
from modules.mlx_planner import propose_geometry


def plan(state):
    print("Plan stage")

    config = None

    # 1. Attempt LLM proposal
    try:
        config = propose_geometry(
            history=state.history,
            best_mean=state.best_mean,
            rejected=state.rejected,          # list of RejectedExperiment
            epistemic_flags=state.epistemic_flags,
            mode=state.mode,
            best_geometry=state.best_geometry,
        )
    except Exception as e:
        print(f"  LLM proposal failed: {e}")
        config = None

    # 2. Fallback if LLM returned nothing or invalid
    if config is None or not isinstance(config, dict):
        config = _fallback_proposal(state)

    # 3. Validate and clamp to valid ranges
    if state.mode == "phase3":
        pt_cut = config.get("pt_cut", 20.0)
        eta_cut = config.get("eta_cut", 2.4)
        iso_cut = config.get("iso_cut", 0.25)
        pt_cut = np.clip(pt_cut, 5.0, 50.0)
        eta_cut = np.clip(eta_cut, 1.5, 3.0)
        iso_cut = np.clip(iso_cut, 0.1, 0.5)
        config["pt_cut"] = float(pt_cut)
        config["eta_cut"] = float(eta_cut)
        config["iso_cut"] = float(iso_cut)
    else:
        # Phase 2: size + material
        size = config.get("detector_size_cm", 12.0)
        material = config.get("material", "G4_WATER")
        size = np.clip(size, 8.0, 20.0)
        if material not in ["G4_WATER", "G4_Si", "G4_Fe", "G4_Al", "G4_Cu", "G4_Pb"]:
            material = "G4_WATER"
        config["detector_size_cm"] = float(size)
        config["material"] = material

    state.geometry = config
    print(f"  Proposed geometry: {config}")
    return state


def _fallback_proposal(state):
    if state.best_geometry is not None:
        base = dict(state.best_geometry)
        # Remove any Phase 2 keys if mode is phase4 or phase3
        if state.mode in ["phase3", "phase4"]:
            base.pop("detector_size_cm", None)
            base.pop("material", None)
        if state.mode == "phase3":
            base["pt_cut"] = np.clip(base.get("pt_cut", 20.0) + np.random.uniform(-2.0, 2.0), 5, 50)
            base["eta_cut"] = np.clip(base.get("eta_cut", 2.4) + np.random.uniform(-0.1, 0.1), 1.5, 3.0)
            base["iso_cut"] = np.clip(base.get("iso_cut", 0.25) + np.random.uniform(-0.02, 0.02), 0.1, 0.5)
        elif state.mode == "phase4":
            base["target_radius"] = np.clip(base.get("target_radius", 30.0) + np.random.uniform(-2.0, 2.0), 10, 50)
            base["target_length"] = np.clip(base.get("target_length", 60.0) + np.random.uniform(-2.0, 2.0), 20, 100)
            base["absorber_thickness"] = np.clip(base.get("absorber_thickness", 2.5) + np.random.uniform(-0.2, 0.2), 0.5, 5)
            base["gap_thickness"] = np.clip(base.get("gap_thickness", 0.5) + np.random.uniform(-0.05, 0.05), 0.1, 1)
            base["field_strength"] = np.clip(base.get("field_strength", 1.0) + np.random.uniform(-0.1, 0.1), 0, 2)
            base["n_layers"] = int(np.clip(base.get("n_layers", 15) + np.random.randint(-2, 3), 5, 30))
            if np.random.rand() < 0.3:
                base["absorber_material"] = np.random.choice(["Pb", "W", "Cu"])
            if np.random.rand() < 0.3:
                base["readout_segmentation"] = np.random.choice([8, 16, 32])
        else:
            # phase2
            base["detector_size_cm"] = np.clip(base.get("detector_size_cm", 12.0) + np.random.uniform(-1.0, 1.0), 8, 20)
            base["material"] = np.random.choice(["G4_WATER","G4_Si","G4_Fe","G4_Al","G4_Cu","G4_Pb"])
        return base
    else:
        # Random proposal
        if state.mode == "phase3":
            return {"pt_cut": float(np.random.uniform(5, 50)), "eta_cut": float(np.random.uniform(1.5, 3.0)), "iso_cut": float(np.random.uniform(0.1, 0.5))}
        elif state.mode == "phase4":
            return {
                "target_radius": float(np.random.uniform(10, 50)),
                "target_length": float(np.random.uniform(20, 100)),
                "absorber_thickness": float(np.random.uniform(0.5, 5)),
                "absorber_material": np.random.choice(["Pb", "W", "Cu"]),
                "gap_thickness": float(np.random.uniform(0.1, 1)),
                "n_layers": int(np.random.randint(5, 31)),
                "field_strength": float(np.random.uniform(0, 2)),
                "readout_segmentation": int(np.random.choice([8, 16, 32])),
            }
        else:
            return {"detector_size_cm": float(np.random.uniform(8, 20)), "material": np.random.choice(["G4_WATER","G4_Si","G4_Fe","G4_Al","G4_Cu","G4_Pb"])}