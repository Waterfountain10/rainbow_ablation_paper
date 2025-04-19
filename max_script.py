import itertools
import subprocess

FLAGS = ["-useDouble", "-usePrioritized", "-useDistributive", "-useDuel", "-useNstep", "-useNoisy"]

def run_experiment(flags, ablation=False):
    cmd = [
        "python3.10", "main.py", # "python3.10" is done for server, change to "python" for local
        *flags,            
    ]
    if ablation:
        cmd.append("-ablation")
        
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # Run base models
    for flag_pair in itertools.combinations(FLAGS, 1):
        run_experiment(flag_pair)
        
    run_experiment([]) # Run without any flags (DQN)
        
    # Run ablation models
    for flag_pair in itertools.combinations(FLAGS, 5):
        run_experiment(flag_pair, ablation=True)
        
    run_experiment(FLAGS, ablation=True) # Run with all flags (Rainbow) (ablation to diff from DQN)
    