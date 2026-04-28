import random
import numpy as np
from modules.mlx_planner import propose_geometry


def plan(state):

    print("Plan stage")
    tested = {
        (h["geometry"]["detector_size_cm"], h["geometry"]["material"])
        for h in state.history
        }
    sizes = [round(random.uniform(8, 20), 2) for _ in range(30)]
    materials = ["G4_WATER","G4_Si","G4_Fe","G4_Al","G4_Cu","G4_Pb"]

    space = [
    (float(s), m)
    for s in sizes
    for m in materials
    ]
    remaining = [
        (s, m)
        for (s, m) in space
        if (round(s, 2), m) not in tested
    ]
    if remaining:
        size, mat = random.choice(remaining)
        config = {"detector_size_cm": size, "material": mat}
    else:
        config = propose_geometry(state.history, state.best_mean)    
    
    state.geometry = config

    print("Proposed geometry:", config)

    return state