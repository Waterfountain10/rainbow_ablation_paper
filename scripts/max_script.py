import itertools
import subprocess

FLAGS = ["-useDouble", "-usePrioritized",
         "-useDistributive", "-useDuel", "-useNstep", "-useNoisy"]

USER = "WILLIAM"  # change to your name


def run_experiment(flags, ablation=False):
    cmd = [
        "python3.10", "main.py",  # "python3.10" is done for server, change to "python" for local
        *flags,
    ]
    if ablation:
        cmd.append("-ablation")
    match USER :
        case "MAX":
            cmd.append("-env")
            cmd.append("ALE/Asterix-ram-v5")
        case "WILLIAM":
            cmd.append("-env")
            cmd.append("ALE/RoadRunner-ram-v5")
        case "DENIS":
            cmd.append("-env")
            cmd.append("ALE/Seaquest-ram-v5")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # Run ablation models
    for flag_pair in itertools.combinations(FLAGS, 5):
        run_experiment(flag_pair, ablation=True)

    # Run with all flags (Rainbow) (ablation to diff from DQN)
    run_experiment(FLAGS, ablation=True)
