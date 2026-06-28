import json
import sys
from geant_runner import run_simulation

def main():
    if len(sys.argv) < 2:
        print("Usage: python replay.py <config.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = json.load(f)

    result = run_simulation(config)
    # Output as JSON for easy parsing
    print(json.dumps(result))

if __name__ == "__main__":
    main()