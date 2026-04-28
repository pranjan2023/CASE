import hashlib
from mlx_lm import load, generate

MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"

PROMPT = """
Explain in one sentence what a Monte Carlo particle transport simulation is.
"""

RUNS = 10


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():

    print("Loading MLX model...")
    model, tokenizer = load(MODEL_NAME)

    hashes = []

    for i in range(RUNS):

        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=PROMPT,
            max_tokens=50,
        )

        text =str(output).strip()

        h = sha256(text)
        hashes.append(h)

        print(f"Run {i+1}:")
        print(text)
        print("SHA256:", h)
        print("-" * 60)

    unique_hashes = set(hashes)

    print("\nHash Summary")
    print("=" * 60)
    for h in hashes:
        print(h)

    assert len(unique_hashes) == 1, "❌ Determinism test FAILED: outputs differ"

    print("\n✅ Determinism verified: all outputs identical")


if __name__ == "__main__":
    main()
