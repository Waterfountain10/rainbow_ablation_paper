import sys
import subprocess
import itertools
from multiprocessing import Pool
import os

def run_ddqn(params):
    lr, = params
    command = [
        sys.executable,  # ← ensures you use your venv’s Python
        "main.py",
        "-memory_size=50000",
        "-lr", str(lr),
        "-useDistributive",
        "-batch_size=64",
        "-v_max=10000",
        "-env=ALE/Asterix-ram-v5",
        "-v_min=-10",
        "-v_max=10000",
        "-atom_size=51",
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

    lr = [2.5e-4, 1e-4, 5e-4, 1e-5, 5e-5]
    param_combinations = list(itertools.product(lr))

    with Pool(processes=5) as pool:
        results = pool.map(run_ddqn, param_combinations)

    print("\n--- Experiment Summary ---")
    for r in results:
        print(r)
    print("--- All experiments finished ---")