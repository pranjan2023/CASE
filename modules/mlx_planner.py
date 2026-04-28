import json
from mlx_lm import generate

model = None
tokenizer = None


def set_model(m, t):
    global model, tokenizer
    model = m
    tokenizer = t


def propose_geometry(history, best_score):

    # fallback if model not loaded
    if model is None or tokenizer is None:
        return {"detector_size_cm": 12.0, "material": "G4_WATER"}

    # format last few experiments
    formatted = []
    for h in history[-5:]:
        formatted.append(
            f"{h['geometry']} → {h['mean']:.3f} ± {h['std']:.3f}"
        )

    history_str = "\n".join(formatted)

    # prompt
    prompt = f"""
You are optimizing a physics detector.

History:
{history_str}

Goal:
maximize mean while considering uncertainty.

Search space:
- detector_size_cm: continuous float in [8, 20]
- material: ["G4_WATER","G4_Si","G4_Fe","G4_Al","G4_Cu","G4_Pb"]

Rules:
- prefer higher mean
- avoid high std unless justified
- explore near best configs

Output JSON:
{{"detector_size_cm": float, "material": str}}
"""

    # generate
    output = generate(model, tokenizer, prompt=prompt, max_tokens=50)

    # parse
    try:
        result = json.loads(output.strip())
    except:
        result = {"detector_size_cm": 12.0, "material": "G4_WATER"}

    # validate + normalize
    allowed_materials = ["G4_WATER","G4_Si","G4_Fe","G4_Al","G4_Cu","G4_Pb"]

    size = result.get("detector_size_cm", 12.0)
    try:
        size = float(size)
    except:
        size = 12.0

    # clamp + normalize
    size = max(8.0, min(20.0, size))
    size = round(size, 2)

    material = result.get("material", "G4_WATER")
    if material not in allowed_materials:
        material = "G4_WATER"

    return {
        "detector_size_cm": size,
        "material": material
    }