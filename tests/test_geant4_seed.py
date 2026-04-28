
import subprocess

seed = 1234


def run():

    result = subprocess.check_output(
        ["python", "run_geant4_seed_sim.py", str(seed)]
    ).decode()

    # Extract last line that contains a number
    lines = result.strip().split("\n")

    for line in reversed(lines):
        try:
            return float(line)
        except ValueError:
            continue

    raise RuntimeError("No numeric output found from simulation")


e1 = run()
e2 = run()

print("Run1:", e1)
print("Run2:", e2)

assert abs(e1 - e2) < 1e-6

print("PASS: Geant4 reproducible")
