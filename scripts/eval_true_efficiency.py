#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uproot
import awkward as ak
import numpy as np
from phase3.hzz_oracle import get_events

def true_efficiency(pt_cut, eta_cut, iso_cut):
    arrays = get_events()
    muon_px = arrays["Muon_Px"]
    muon_py = arrays["Muon_Py"]
    muon_pz = arrays["Muon_Pz"]
    muon_iso = arrays["Muon_Iso"]

    pt_per = np.sqrt(muon_px**2 + muon_py**2)
    pt_safe = ak.where(pt_per > 0, pt_per, 1e-9)
    eta_per = np.arcsinh(muon_pz / pt_safe)

    pass_pt  = pt_per > pt_cut
    pass_eta = np.abs(eta_per) < eta_cut
    pass_iso = muon_iso < iso_cut

    pass_all = pass_pt & pass_eta & pass_iso
    event_pass = ak.any(pass_all, axis=1)
    return float(ak.mean(event_pass))

if __name__ == "__main__":
    # Test with loose cuts
    print("Loose cuts (pt=5, eta=3, iso=0.5):", true_efficiency(5.0, 3.0, 0.5))