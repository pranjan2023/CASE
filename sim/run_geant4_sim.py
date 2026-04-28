import sys
import json
from geant_runner import run_simulation

config = json.loads(sys.argv[1])

results = run_simulation(config)

print(json.dumps(results))
