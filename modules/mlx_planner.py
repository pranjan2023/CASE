import json
import random
import numpy as np
from mlx_lm import generate

model = None
tokenizer = None


def set_model(m, t):
    global model, tokenizer
    model = m
    tokenizer = t


def propose_geometry(history, best_mean=None, rejected=None, epistemic_flags=None,
                     mode="phase2", best_geometry=None):
    if model is not None and tokenizer is not None:
        try:
            result = _llm_proposal(history, best_mean, rejected, epistemic_flags, mode)
            if result is not None:
                return result
        except Exception as e:
            print(f"LLM proposal failed, falling back to heuristic: {e}")
    return _heuristic_proposal(history, best_mean, best_geometry, mode)


def _llm_proposal(history, best_mean, rejected, epistemic_flags, mode):
    # Build prompt (same as before)
    formatted_history = []
    for h in history[-5:]:
        geom = h.get("geometry", {})
        mean = h.get("mean", 0.0)
        std = h.get("std", 0.0)
        accepted = h.get("accepted", False)
        formatted_history.append(
            f"{geom} → {mean:.4f} ± {std:.4f}  {'ACCEPTED' if accepted else 'REJECTED'}"
        )
    history_str = "\n".join(formatted_history) if formatted_history else "(no history yet)"

    rejection_str = ""
    if rejected:
        recent = rejected[-5:]
        lines = []
        for r in recent:
            geom = r.geometry
            reason = r.rejection_reason
            mean = r.mean
            best_at = r.best_mean_at_rejection
            ad_stat = r.cross_physics_stat
            lines.append(
                f"Rejected: {geom}  → mean={mean:.4f}, reason={reason}, best_at_time={best_at:.4f}" +
                (f", AD_stat={ad_stat:.4f}" if ad_stat else "")
            )
        rejection_str = "Recent rejections:\n" + "\n".join(lines) + "\n"

    epi_str = ""
    if epistemic_flags:
        latest = epistemic_flags[-1]
        epi_str = (
            f"Latest epistemic: cross_physics_consistent={latest.cross_physics_consistent}, "
            f"AD_stat={latest.anderson_darling_stat or 0.0:.4f}, "
            f"jackknife_std={latest.jackknife_std or 0.0:.4f}, "
            f"rejection_reason={latest.rejection_reason or 'None'}"
        )

    best_str = f"Best so far: mean={best_mean:.4f}" if best_mean is not None else "No best yet."

    if mode == "phase3":
        search_space = """
- pt_cut   : float in [5, 50]   GeV
- eta_cut  : float in [1.5, 3.0]
- iso_cut  : float in [0.1, 0.5]
"""
        output_format = '{"pt_cut": float, "eta_cut": float, "iso_cut": float}'
    else:
        search_space = """
- detector_size_cm : float in [8, 20]
- material         : one of ["G4_WATER","G4_Si","G4_Fe","G4_Al","G4_Cu","G4_Pb"]
"""
        output_format = '{"detector_size_cm": float, "material": str}'

    prompt = f"""
You are an AI planner optimizing a physics detector.

Current best: {best_str}
{rejection_str}
{epi_str}

Recent history (last 5):
{history_str}

Search space:
{search_space}

Rules:
- Propose a new configuration that improves upon the best, avoiding previous rejection reasons.
- Prefer configurations with higher mean and lower uncertainty.
- Explore regions around the best but be cautious of rejections.
- Output only valid JSON with the required keys. Do not include any other text.

Output JSON:
{output_format}
"""

    try:
        output = generate(model, tokenizer, prompt=prompt, max_tokens=100)
        # Extract first JSON object using regex or manual search
        start = output.find('{')
        if start == -1:
            return None
        # Find the matching closing brace by counting
        brace_count = 0
        end = start
        for i, ch in enumerate(output[start:], start=start):
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        if brace_count != 0:
            return None
        json_str = output[start:end]
        result = json.loads(json_str)
    except Exception as e:
        print(f"LLM generation/parse error: {e}")
        return None

    return _validate_proposal(result, mode)


def _validate_proposal(result, mode):
    if mode == "phase3":
        # existing phase3 validation
        pt = result.get("pt_cut", 20.0)
        eta = result.get("eta_cut", 2.4)
        iso = result.get("iso_cut", 0.25)
        try:
            pt = float(pt)
        except:
            pt = 20.0
        try:
            eta = float(eta)
        except:
            eta = 2.4
        try:
            iso = float(iso)
        except:
            iso = 0.25
        pt = np.clip(pt, 5.0, 50.0)
        eta = np.clip(eta, 1.5, 3.0)
        iso = np.clip(iso, 0.1, 0.5)
        return {
            "pt_cut": round(pt, 2),
            "eta_cut": round(eta, 3),
            "iso_cut": round(iso, 3)
        }
    elif mode == "phase4":
        # Phase 4 validation
        defaults = {
            "target_radius": 30.0,
            "target_length": 60.0,
            "absorber_thickness": 2.5,
            "absorber_material": "Pb",
            "gap_thickness": 0.5,
            "n_layers": 15,
            "field_strength": 1.0,
            "readout_segmentation": 16,
        }
        validated = {}
        for k, v in defaults.items():
            val = result.get(k, v)
            if k == "target_radius":
                val = float(val); val = np.clip(val, 10, 50); val = round(val, 2)
            elif k == "target_length":
                val = float(val); val = np.clip(val, 20, 100); val = round(val, 2)
            elif k == "absorber_thickness":
                val = float(val); val = np.clip(val, 0.5, 5); val = round(val, 3)
            elif k == "absorber_material":
                if val not in ["Pb", "W", "Cu"]:
                    val = "Pb"
            elif k == "gap_thickness":
                val = float(val); val = np.clip(val, 0.1, 1); val = round(val, 3)
            elif k == "n_layers":
                val = int(val); val = int(np.clip(val, 5, 30))
            elif k == "field_strength":
                val = float(val); val = np.clip(val, 0, 2); val = round(val, 2)
            elif k == "readout_segmentation":
                if val not in [8, 16, 32]:
                    val = 16
            validated[k] = val
        return validated
    else:
        # fallback for phase2 or others
        size = result.get("detector_size_cm", 12.0)
        mat = result.get("material", "G4_WATER")
        try:
            size = float(size)
        except:
            size = 12.0
        size = np.clip(size, 8.0, 20.0)
        allowed = ["G4_WATER","G4_Si","G4_Fe","G4_Al","G4_Cu","G4_Pb"]
        if mat not in allowed:
            mat = "G4_WATER"
        return {
            "detector_size_cm": round(size, 2),
            "material": mat
        }


def _heuristic_proposal(history, best_mean, best_geometry, mode):
    """Heuristic with mutation and persistent random search."""
    tested = set()
    for h in history:
        geom = h.get("geometry", {})
        if mode == "phase3":
            key = (geom.get("pt_cut"), geom.get("eta_cut"), geom.get("iso_cut"))
        else:
            # For phase2 or others (maybe phase4), use a generic key
            # but we'll handle phase3 and fallback to a generic tuple
            key = tuple(sorted(geom.items()))
        tested.add(key)

    if mode == "phase3":
        # ----- Phase 3: pt_cut, eta_cut, iso_cut -----
        if best_geometry is not None:
            for attempt in range(50):
                pt = best_geometry.get("pt_cut", 20.0) + np.random.normal(0, 2.0 * (1 + attempt*0.1))
                eta = best_geometry.get("eta_cut", 2.4) + np.random.normal(0, 0.1 * (1 + attempt*0.1))
                iso = best_geometry.get("iso_cut", 0.25) + np.random.normal(0, 0.02 * (1 + attempt*0.1))
                pt = np.clip(pt, 5.0, 50.0)
                eta = np.clip(eta, 1.5, 3.0)
                iso = np.clip(iso, 0.1, 0.5)
                new_geom = {
                    "pt_cut": round(pt, 2),
                    "eta_cut": round(eta, 3),
                    "iso_cut": round(iso, 3)
                }
                key = (new_geom["pt_cut"], new_geom["eta_cut"], new_geom["iso_cut"])
                if key not in tested:
                    return new_geom

        # If no best or mutation failed, random untested
        for _ in range(500):
            pt = np.random.uniform(5.0, 50.0)
            eta = np.random.uniform(1.5, 3.0)
            iso = np.random.uniform(0.1, 0.5)
            new_geom = {
                "pt_cut": round(pt, 2),
                "eta_cut": round(eta, 3),
                "iso_cut": round(iso, 3)
            }
            key = (new_geom["pt_cut"], new_geom["eta_cut"], new_geom["iso_cut"])
            if key not in tested:
                return new_geom

        # Ultimate fallback for phase3
        return {"pt_cut": 20.0, "eta_cut": 2.4, "iso_cut": 0.25}

    else:
        # ----- Phase 2 / Phase 4 / other -----
        # (This is the existing Phase 4 grammar; keep it for future use)
        if best_geometry is not None:
            for attempt in range(50):
                new_geom = best_geometry.copy()
                # Mutate continuous variables
                new_geom["target_radius"] = np.clip(
                    new_geom.get("target_radius", 30) + np.random.normal(0, 2.0 * (1+attempt*0.1)), 10, 50
                )
                new_geom["target_length"] = np.clip(
                    new_geom.get("target_length", 60) + np.random.normal(0, 2.0 * (1+attempt*0.1)), 20, 100
                )
                new_geom["absorber_thickness"] = np.clip(
                    new_geom.get("absorber_thickness", 2.5) + np.random.normal(0, 0.2 * (1+attempt*0.1)), 0.5, 5
                )
                new_geom["gap_thickness"] = np.clip(
                    new_geom.get("gap_thickness", 0.5) + np.random.normal(0, 0.05 * (1+attempt*0.1)), 0.1, 1
                )
                new_geom["field_strength"] = np.clip(
                    new_geom.get("field_strength", 1.0) + np.random.normal(0, 0.1 * (1+attempt*0.1)), 0, 2
                )
                # Integer variables
                new_geom["n_layers"] = int(np.clip(
                    new_geom.get("n_layers", 15) + np.random.poisson(1) - 1, 5, 30
                ))
                # Discrete
                if np.random.rand() < 0.3:
                    new_geom["absorber_material"] = np.random.choice(["Pb", "W", "Cu"])
                if np.random.rand() < 0.3:
                    new_geom["readout_segmentation"] = np.random.choice([8, 16, 32])
                key = tuple(sorted(new_geom.items()))
                if key not in tested:
                    return new_geom

        # Random untested for other modes
        for _ in range(500):
            new_geom = {
                "target_radius": np.random.uniform(10, 50),
                "target_length": np.random.uniform(20, 100),
                "absorber_thickness": np.random.uniform(0.5, 5),
                "absorber_material": np.random.choice(["Pb", "W", "Cu"]),
                "gap_thickness": np.random.uniform(0.1, 1),
                "n_layers": np.random.randint(5, 31),
                "field_strength": np.random.uniform(0, 2),
                "readout_segmentation": np.random.choice([8, 16, 32]),
            }
            key = tuple(sorted(new_geom.items()))
            if key not in tested:
                return new_geom

        # Ultimate fallback for phase4
        return {
            "target_radius": 30,
            "target_length": 60,
            "absorber_thickness": 2.5,
            "absorber_material": "Pb",
            "gap_thickness": 0.5,
            "n_layers": 15,
            "field_strength": 1.0,
            "readout_segmentation": 16,
        }