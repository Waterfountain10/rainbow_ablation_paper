import itertools
import subprocess

FLAGS = ["-useDouble", "-usePrioritized", "-useDistributive", "-useDuel", "-useNstep"]

def run_experiment(flags):
    cmd = [
        "python3.10", "main.py",
        *flags,               # unpack the two flags
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    for flag_pair in itertools.combinations(FLAGS, 1):
        run_experiment(flag_pair)
