"""
execute.py — routes to Geant4 (Phase 2), HZZ (Phase 3), mock (Phase 3 mock), or Phase 4.
"""

import subprocess
import json
import numpy as np
from core.utils import hash_config


def execute(state):
    mode = getattr(state, "mode", "phase2")
    if mode == "phase4":
        return _execute_phase4(state)
    elif mode == "phase3":
        if getattr(state, "use_mock_oracle", False):
            return _execute_mock(state)
        else:
            return _execute_phase3(state)
    else:  # phase2
        return _execute_phase2(state)


def _execute_phase2(state):
    print("Execute stage — running Geant4 simulation (Phase 2)")
    physics_lists = ["FTFP_BERT", "QGSP_BERT", "QGSP_BIC"]
    results = {}
    base_seed = 42

    for i, pl in enumerate(physics_lists):
        config = dict(state.geometry)
        if "material" not in config:
            config["material"] = "G4_WATER"
        config["physics_list"] = pl
        config["seed"] = base_seed + i * 1000
        config_id = hash_config(config)
        config.setdefault("events", 1000)
        config.setdefault("depth_cm", config.get("detector_size_cm", 12.0))
        config["config_id"] = config_id

        # Clean numpy types for JSON serialization
        clean_config = json.loads(
            json.dumps(
                config,
                default=lambda x: int(x) if isinstance(x, (np.integer, np.int64))
                else float(x) if isinstance(x, (np.floating, np.float64))
                else str(x) if isinstance(x, (np.str_, np.bytes_))
                else x
            )
        )

        out = subprocess.check_output(
            ["python", "sim/run_geant4_sim.py", json.dumps(clean_config)]
        ).decode()

        for line in reversed(out.split("\n")):
            try:
                data = json.loads(line)
                results[pl] = {"config_id": config_id, **data}
                break
            except:
                continue

    state.results = results
    return state


def _execute_phase3(state):
    from phase3.hzz_oracle import run_oracle
    print("Execute stage — running HZZ reconstruction efficiency oracle (Phase 3)")
    physics_lists = ["FTFP_BERT", "QGSP_BERT", "QGSP_BIC"]
    results = {}
    base_seed = 42
    geom = state.geometry
    geom.setdefault("pt_cut", 20.0)
    geom.setdefault("eta_cut", 2.4)
    geom.setdefault("iso_cut", 0.25)
    runs = getattr(state, "oracle_runs", 20)
    events = getattr(state, "oracle_events", 1000)
    for i, pl in enumerate(physics_lists):
        config = dict(geom)
        config["seed"] = base_seed + i * 1000
        config["runs"] = runs
        config["events"] = events
        out = run_oracle(config)
        results[pl] = {
            "scores": out["scores"],
            "config_id": hash_config(config),
        }
    state.results = results
    return state


def _execute_mock(state):
    from phase3.mock_oracle import run_oracle
    print("Execute stage — running MOCK oracle")
    physics_lists = ["FTFP_BERT", "QGSP_BERT", "QGSP_BIC"]
    results = {}
    base_seed = 42
    geom = state.geometry
    geom.setdefault("pt_cut", 20.0)
    geom.setdefault("eta_cut", 2.4)
    geom.setdefault("iso_cut", 0.25)
    runs = getattr(state, "oracle_runs", 20)
    events = getattr(state, "oracle_events", 1000)
    for i, pl in enumerate(physics_lists):
        config = dict(geom)
        config["seed"] = base_seed + i * 1000
        config["runs"] = runs
        config["events"] = events
        out = run_oracle(config)
        results[pl] = {
            "scores": out["scores"],
            "config_id": hash_config(config),
        }
    state.results = results
    return state


def _execute_phase4(state):
    from phase4.mock_phase4_oracle import run_oracle
    print("Execute stage — running Phase 4 mock oracle")
    physics_lists = ["FTFP_BERT", "QGSP_BERT", "QGSP_BIC"]
    results = {}
    base_seed = 42
    geom = state.geometry
    defaults = {
        "target_radius": 30,
        "target_length": 60,
        "absorber_thickness": 2.5,
        "absorber_material": "Pb",
        "gap_thickness": 0.5,
        "n_layers": 15,
        "field_strength": 1.0,
        "readout_segmentation": 16,
    }
    for k, v in defaults.items():
        geom.setdefault(k, v)
    runs = getattr(state, "oracle_runs", 50)
    events = getattr(state, "oracle_events", 2000)
    for i, pl in enumerate(physics_lists):
        config = dict(geom)
        config["seed"] = base_seed + i * 1000
        config["runs"] = runs
        config["events"] = events
        out = run_oracle(config)
        results[pl] = {
            "scores": out["scores"],
            "config_id": hash_config(config),
        }
    state.results = results
    return state