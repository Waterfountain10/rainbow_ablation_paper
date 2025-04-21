import subprocess
import itertools
from multiprocessing import Pool
import os

def run_ddqn(params):
    """Runs the ddqn.py script with the given parameters."""
    memory_size, batch_size, target_update_freq,  lr, omega, beta, gamma, hidden_dim  = params
    command = [
        "python",
        "main.py",
        "-memory_size", str(memory_size),
        "-batch_size", str(batch_size),
        "-target_update_freq", str(target_update_freq),
        "-lr", str(lr),
        "-omega", str(omega),
        "-beta", str(beta),
        "-gamma", str(gamma),
        "-hidden_dim", str(hidden_dim),
        "-useDouble"
        # Add other arguments if needed, e.g., -num_episodes
        # "-num_episodes", "500" # Example: uncomment and set if you want to override the default
    ]
    print(f"Running with params: {params}")
    try:
        # Use subprocess.run which waits for the command to complete
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Finished run with params: {params}")
        # Optional: print stdout or stderr if needed
        # print("STDOUT:", result.stdout)
        # print("STDERR:", result.stderr)
        return f"Success: {params}"
    except subprocess.CalledProcessError as e:
        print(f"Error running with params: {params}")
        print(f"Command: {' '.join(command)}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Stderr: {e.stderr}")
        return f"Failed: {params} - {e}"
    except Exception as e:
        print(f"An unexpected error occurred with params: {params}")
        print(f"Error: {e}")
        return f"Failed: {params} - {e}"

if __name__ == "__main__":
    # Define hyperparameter ranges to test
    memory_sizes = [80000, 100000]
    batch_sizes = [64, 256, 512]
    target_update_freqs = [1000] # Adjust based on ddqn.py defaults/needs
    # epsilon_decay_steps_list = [500000, 1000000] # Adjust based on ddqn.py defaults/needs
    learning_rates = [5e-5, 1e-4]
    # min_epsilons = [0.01, 0.05]
    omegas = [0.6]
    betas = [0.4]
    gammas = [0.99, 0.95]
    hidden_dims = [256, 512]

    # Create all combinations of hyperparameters
    param_combinations = list(itertools.product(
        memory_sizes,
        batch_sizes,
        target_update_freqs,
        learning_rates,
        omegas,
        betas,
        gammas,
        hidden_dims
    ))

    print(f"Total experiments to run: {len(param_combinations)}")

    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)

    # Number of processes to run in parallel
    num_processes = 2

    # Use a Pool to run experiments in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_ddqn, param_combinations)

    print("\n--- Experiment Summary ---")
    for result in results:
        print(result)
    print("--- All experiments finished ---")
