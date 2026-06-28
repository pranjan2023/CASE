import numpy as np

def run_oracle(config):
    # Extract parameters
    x = config.get("target_radius", 30)
    y = config.get("target_length", 60)
    # Rastrigin function (global min at (0,0) but we want max)
    # Invert and add noise
    score = - (10 + (x-15)**2 - 10*np.cos(2*np.pi*(x-15)) + (y-30)**2 - 10*np.cos(2*np.pi*(y-30)))
    # Add discrete effects
    mat_effect = {"Pb": 0.1, "W": -0.05, "Cu": 0.0}[config.get("absorber_material", "Pb")]
    score += mat_effect
    # Simulate runs
    n_runs = config.get("runs", 20)
    scores = np.random.normal(score, 0.1, n_runs).tolist()
    return {"scores": scores, "events": 1000}