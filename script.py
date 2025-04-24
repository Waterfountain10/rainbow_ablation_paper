import sys
import subprocess
import itertools
from multiprocessing import Pool
import os

def run_ddqn(params):
    flags = params
    command = [
        sys.executable,  # ← ensures you use your venv’s Python
        "main.py",
        "-batch_size=64",
        "-env=ALE/Asterix-ram-v5",
        "-v_min=10",
        "-v_max=10000",
        "-atom_size=51",
        *flags,
    ]
    print(f"Running with params: {params}")
    try:
        subprocess.run(command, check=True)
        return f"Success: {params}"
    except subprocess.CalledProcessError as e:
        print(f"Failed {params}: {e}")
        return f"Failed: {params}"

if __name__ == "__main__":
   
    os.makedirs("results", exist_ok=True)
    
    FLAGS = ["-useDouble", "-usePrioritized", "-useDuel", "-useNoisy", "-useDistributive", "-useNstep"]
    param_combinations = list(itertools.combinations(FLAGS, 1))
    param_combinations.append(FLAGS)  # Add the full set of flags
    param_combinations.append([]) # Add the empty set of flags

    with Pool(processes=5) as pool:
        results = pool.map(run_ddqn, param_combinations)

    print("\n--- Experiment Summary ---")
    for r in results:
        print(r)
    print("--- All experiments finished ---")