from mlx_lm import load
from modules.mlx_planner import set_model
from core.state import SystemState
from core.orchestrator import run_cycle


def main():

    print("Loading MLX model...")

    model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")

    set_model(model, tokenizer)

    print("Model loaded")

    state = SystemState()

    for _ in range(5):
        state = run_cycle(state)


if __name__ == "__main__":
    main()
