"""
hzz_oracle.py — Phase 3 execution oracle replacing Geant4.

Dataset: H→ZZ→4l from scikit-hep public ROOT file (uproot).
Observable: reconstruction efficiency = fraction of events where
  at least one muon passes all three cuts:
    pt  > pt_cut   (transverse momentum, GeV)
    |eta| < eta_cut (pseudorapidity acceptance)
    iso < iso_cut  (isolation — lower = stricter)

The agent proposes (pt_cut, eta_cut, iso_cut) instead of
(detector_size_cm, material). Grammar is re-parameterised but
the observe-plan-execute-verify-update loop is identical.

Physics lists → seed offsets (preserves cross-validation structure):
  "FTFP_BERT"  → offset 0
  "QGSP_BERT"  → offset 1000
  "QGSP_BIC"   → offset 2000
"""

import numpy as np
import uproot
import awkward as ak
from pathlib import Path
import urllib.request

HZZ_URL = "https://raw.githubusercontent.com/scikit-hep/scikit-hep-testdata/master/src/skhep_testdata/data/uproot-HZZ.root"
HZZ_CACHE = Path(__file__).parent / "HZZ.root"


def _ensure_dataset():
    if not HZZ_CACHE.exists():
        print(f"[hzz_oracle] Downloading HZZ.root (~300KB)...")
        urllib.request.urlretrieve(HZZ_URL, HZZ_CACHE)
        print(f"[hzz_oracle] Saved to {HZZ_CACHE}")


def _load_events() -> dict:
    _ensure_dataset()
    with uproot.open(str(HZZ_CACHE) + ":events") as tree:
        arrays = tree.arrays(
            ["NMuon", "Muon_Px", "Muon_Py", "Muon_Pz", "Muon_E", "Muon_Iso"],
            library="ak"
        )
    return arrays


# Load once per process
_EVENTS_CACHE = None


def get_events():
    global _EVENTS_CACHE
    if _EVENTS_CACHE is None:
        _EVENTS_CACHE = _load_events()
    return _EVENTS_CACHE


def compute_efficiency(arrays, pt_cut: float, eta_cut: float, iso_cut: float,
                       rng: np.random.Generator, n_events: int = 1000) -> float:
    """
    Efficiency = fraction of sampled events where at least one muon
    passes all three cuts. Uses real HZZ kinematics, no synthetic noise.

    eta computed as arcsinh(pz/pt) — massless limit, valid for muons at LHC.
    """
    n_total = len(arrays["NMuon"])
    idx = rng.choice(n_total, size=min(n_events, n_total), replace=False)

    muon_px  = arrays["Muon_Px"][idx]
    muon_py  = arrays["Muon_Py"][idx]
    muon_pz  = arrays["Muon_Pz"][idx]
    muon_iso = arrays["Muon_Iso"][idx]

    pt_per   = np.sqrt(muon_px**2 + muon_py**2)
    pt_safe  = ak.where(pt_per > 0, pt_per, 1e-9)
    eta_per  = np.arcsinh(muon_pz / pt_safe)

    pass_pt  = pt_per  > pt_cut
    pass_eta = np.abs(eta_per) < eta_cut
    pass_iso = muon_iso < iso_cut

    pass_all   = pass_pt & pass_eta & pass_iso   # jagged bool per muon
    event_pass = ak.any(pass_all, axis=1)         # per-event bool

    return float(ak.mean(event_pass))


def run_oracle(config: dict) -> dict:
    """
    Drop-in replacement for run_simulation() in geant_runner.py.

    Config keys:
      pt_cut   : float  [5, 50]   GeV
      eta_cut  : float  [1.5, 3.0]
      iso_cut  : float  [0.1, 0.5]
      runs     : int    default 20
      events   : int    default 1000
      seed     : int    default 42
    """
    pt_cut  = float(config.get("pt_cut",  20.0))
    eta_cut = float(config.get("eta_cut",  2.4))
    iso_cut = float(config.get("iso_cut",  0.25))
    runs    = int(config.get("runs",  20))
    n_ev    = int(config.get("events", 1000))
    seed    = int(config.get("seed",   42))

    arrays = get_events()

    scores = []
    for i in range(runs):
        sub_rng = np.random.default_rng(seed + i * 37)
        eff = compute_efficiency(arrays, pt_cut, eta_cut, iso_cut, sub_rng, n_ev)
        scores.append(eff)

    return {
        "events": n_ev,
        "scores": scores,
    }