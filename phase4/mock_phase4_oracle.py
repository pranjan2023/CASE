# phase4/mock_phase4_oracle.py
import numpy as np

def run_oracle(config):
    """
    Mock oracle for Phase 4: Rastrigin-like function with discrete parameters.
    Optimum: target_radius=25, target_length=50, absorber_thickness=2,
             absorber_material="Pb", gap_thickness=0.5, n_layers=15,
             field_strength=1.0, readout_segmentation=16.
    Score = -sum of Rastrigin terms (so we maximize).
    """
    # Material penalty: Pb=0, W=1, Cu=2
    mat_map = {"Pb": 0.0, "W": 1.0, "Cu": 2.0}
    mat = config.get("absorber_material", "Pb")
    mat_num = mat_map.get(mat, 0.0)

    # Extract parameters with defaults
    r = config.get("target_radius", 30.0)
    L = config.get("target_length", 60.0)
    t_abs = config.get("absorber_thickness", 2.5)
    g = config.get("gap_thickness", 0.5)
    n = config.get("n_layers", 15)
    B = config.get("field_strength", 1.0)
    seg = config.get("readout_segmentation", 16)

    # Known optimum values
    opt_r = 25
    opt_L = 50
    opt_t = 2.0
    opt_g = 0.5
    opt_n = 15
    opt_B = 1.0
    opt_seg = 16

    # Rastrigin terms (each term = 10 + (x-opt)^2 - 10*cos(2π*(x-opt)))
    score = 0.0
    score += 10 + (r - opt_r)**2 - 10 * np.cos(2 * np.pi * (r - opt_r))
    score += 10 + (L - opt_L)**2 - 10 * np.cos(2 * np.pi * (L - opt_L))
    score += 10 + (t_abs - opt_t)**2 - 10 * np.cos(2 * np.pi * (t_abs - opt_t))
    score += 10 + (g - opt_g)**2 - 10 * np.cos(2 * np.pi * (g - opt_g))
    score += 10 + (n - opt_n)**2 - 10 * np.cos(2 * np.pi * (n - opt_n))
    score += 10 + (B - opt_B)**2 - 10 * np.cos(2 * np.pi * (B - opt_B))
    score += 10 + (seg - opt_seg)**2 - 10 * np.cos(2 * np.pi * (seg - opt_seg))
    # Add material penalty
    score += mat_num * 0.5

    # We want to maximize, so return negative score
    final_score = -score

    # Add noise
    n_runs = config.get("runs", 20)
    scores = np.random.normal(final_score, 0.5, n_runs).tolist()
    return {"scores": scores, "events": 1000}