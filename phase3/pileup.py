"""
pileup.py — Pile-up overlay for Phase 3 HZZ oracle.

Models 200 sub-event pile-up (µ=200) by:
- Randomly sampling additional events from the dataset.
- Adding their isolation contributions (Muon_Iso) as a background term.
- Re-evaluating the isolation cut after overlay.

This is a realistic stress test for the FDR gatekeeping.
"""

import numpy as np
import awkward as ak


def add_pileup(arrays, mu: int = 200, rng=None):
    """
    Overlay pile-up events onto the original array.

    Returns a new array with:
      - Muon_Px, Py, Pz, E unchanged (signal)
      - Muon_Iso increased by sum of isolation from pile-up muons
    """
    if rng is None:
        rng = np.random.default_rng()

    n_total = len(arrays["NMuon"])
    if n_total == 0:
        return arrays

    # Number of pile-up events per event: Poisson(mu)
    n_pileup_events = rng.poisson(mu, size=n_total)
    max_pileup = int(np.max(n_pileup_events))

    if max_pileup == 0:
        return arrays

    # Sample pile-up muon indices from the whole dataset
    pileup_indices = rng.choice(n_total, size=(n_total, max_pileup))
    # For events with fewer pileups, we'll mask later

    # Get isolation for pile-up muons
    pileup_iso = arrays["Muon_Iso"][pileup_indices]  # shape (n_total, max_pileup)
    # Mask out extra pileups beyond the Poisson draw
    mask = ak.Array([np.arange(max_pileup) < n for n in n_pileup_events])
    pileup_iso_masked = ak.where(mask, pileup_iso, 0.0)

    # Sum isolation contributions per event
    pileup_sum = ak.sum(pileup_iso_masked, axis=1)

    # Add to original isolation
    new_iso = arrays["Muon_Iso"] + pileup_sum

    # Return a new dictionary with modified isolation
    new_arrays = dict(arrays)
    new_arrays["Muon_Iso"] = new_iso
    return new_arrays