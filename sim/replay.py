import json
import sys
from geant_runner import run_simulation

with open(sys.argv[1]) as f:
    config = json.load(f)

print(run_simulation(config))
