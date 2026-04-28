import subprocess
import json
from core.utils import hash_config



def execute(state):
    print("Execute stage — running simulation")
    physics_lists = ["FTFP_BERT", "QGSP_BERT", "QGSP_BIC"]
    results = {}
    base_seed=42
    for i, pl in enumerate(physics_lists):
        config = dict(state.geometry)   # copy
        if "material" not in config:
            config["material"] = "G4_WATER"

        config["physics_list"] = pl
        config["seed"] = base_seed + i * 1000
        config_id = hash_config(config)
        config.setdefault("events", 1000)
        config.setdefault("depth_cm", config["detector_size_cm"])
        config["config_id"] = config_id
        out = subprocess.check_output(
            [
                "python",
                "sim/run_geant4_sim.py",
                json.dumps(config)
            ]
        ).decode()
        for line in reversed(out.split("\n")):
            try:
                results[pl] = {"config_id": config_id,**json.loads(line)}
                break
            except:
                continue
    state.results = results
    return state