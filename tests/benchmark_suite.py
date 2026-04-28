import random
import os
import matplotlib.pyplot as plt
import numpy as np

from core.orchestrator import run_cycle, run_fixed
from core.state import SystemState
from verification.scoring import p_value

# -------------------------
# CONFIG SPACE (EXPANDED)
# -------------------------
SPACE = [
    {"detector_size_cm": float(s), "material": m}
    for s in np.linspace(8, 20, 13)
    for m in ["G4_WATER","G4_Si","G4_Fe","G4_Al","G4_Cu","G4_Pb"]
]

OUT_DIR = "benchmark_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------
# CORE RUN (NO PLANNER)
# -------------------------
def run_once(config):
    state = SystemState()
    state.geometry = config
    state = run_fixed(state)
    return state.last_mean


# -------------------------
# BENCHMARK 1: EFFICIENCY
# -------------------------
def benchmark_efficiency(max_iters=18):
    # CASE (planner ON)
    state = SystemState()
    case_curve = []

    for _ in range(max_iters):
        state = run_cycle(state)
        case_curve.append(state.best_mean)

    # RANDOM (same budget)
    random_curve = []
    best = 0.0
    sampled = random.sample(SPACE, max_iters)

    for cfg in sampled:
        score = run_once(cfg)
        best = max(best, score)
        random_curve.append(best)

    return {
        "case": case_curve,
        "random": random_curve
    }


def plot_efficiency(res):
    plt.figure()
    plt.plot(res["case"], label="CASE")
    plt.plot(res["random"], label="Random")
    plt.xlabel("Iteration")
    plt.ylabel("Best TAR")
    plt.legend()
    plt.savefig(f"{OUT_DIR}/efficiency.png")
    plt.close()


# -------------------------
# BENCHMARK 2: FDR
# -------------------------
def benchmark_fdr():
    accepted_fdr = []
    accepted_no_fdr = []

    best = None

    for cfg in SPACE:
        state = SystemState()
        state.geometry = cfg
        state = run_fixed(state)  # NO planner

        mean = state.last_mean

        if best is None:
            best = mean
            continue

        is_true = mean > best

        p = p_value(
            {"mean": mean, "std": state.last_std},
            {"mean": best, "std": 0.01}
        )

        # naive acceptance
        if mean > best:
            accepted_no_fdr.append(is_true)

        # FDR acceptance
        if p < 0.05:
            accepted_fdr.append(is_true)

        best = max(best, mean)

    return {
        "fdr_precision": (
            sum(accepted_fdr) / len(accepted_fdr)
            if accepted_fdr else 0
        ),
        "no_fdr_precision": (
            sum(accepted_no_fdr) / len(accepted_no_fdr)
            if accepted_no_fdr else 0
        )
    }


def plot_fdr(res):
    labels = ["FDR", "No FDR"]
    values = [res["fdr_precision"], res["no_fdr_precision"]]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Precision")
    plt.savefig(f"{OUT_DIR}/fdr.png")
    plt.close()


# -------------------------
# BENCHMARK 3: CROSS-PHYSICS
# -------------------------
def benchmark_cross_physics():
    bert = []
    qgsp = []

    for cfg in SPACE:
        state = SystemState()
        state.geometry = cfg
        state = run_fixed(state)  # NO planner

        stats = state.multi_stats

        bert.append(stats["FTFP_BERT"]["mean"])
        qgsp.append(stats["QGSP_BERT"]["mean"])

    corr = np.corrcoef(bert, qgsp)[0, 1]
    diffs = [abs(a - b) for a, b in zip(bert, qgsp)]

    return {
        "correlation": corr,
        "mean_diff": np.mean(diffs),
        "max_diff": np.max(diffs),
        "bert": bert,
        "qgsp": qgsp
    }


def plot_cross(res):
    plt.figure()
    plt.scatter(res["bert"], res["qgsp"])
    plt.xlabel("FTFP_BERT")
    plt.ylabel("QGSP_BERT")
    plt.savefig(f"{OUT_DIR}/cross_physics.png")
    plt.close()


# -------------------------
# DIVERGENCE SIGNAL
# -------------------------
def divergence_signal():
    data = []

    for cfg in SPACE:
        state = SystemState()
        state.geometry = cfg
        state = run_fixed(state)  # NO planner

        stats = state.multi_stats

        m1 = stats["FTFP_BERT"]["mean"]
        m2 = stats["QGSP_BERT"]["mean"]

        diff = abs(m1 - m2)
        quality = (m1 + m2) / 2

        data.append((diff, quality))

    return data


def plot_divergence(data):
    x = [d[0] for d in data]
    y = [d[1] for d in data]

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("Physics divergence")
    plt.ylabel("Mean TAR")
    plt.savefig(f"{OUT_DIR}/divergence.png")
    plt.close()


# -------------------------
# RUN ALL
# -------------------------
def run_all():
    eff = benchmark_efficiency()
    plot_efficiency(eff)

    fdr = benchmark_fdr()
    plot_fdr(fdr)

    cross = benchmark_cross_physics()
    plot_cross(cross)

    div = divergence_signal()
    plot_divergence(div)

    print("EFFICIENCY:", eff)
    print("FDR:", fdr)
    print("CROSS:", {
        "correlation": cross["correlation"],
        "mean_diff": cross["mean_diff"],
        "max_diff": cross["max_diff"]
    })


if __name__ == "__main__":
    run_all()