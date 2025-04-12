# generate graphs like rainbow paper using results/
import numpy as np
from run import AGENTS

# Helper function for plotting sliding window mean (reduces variance)
def running_mean(x, window_size):
    cumsum = np.cumsum(np.insert(x,0,0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

def plot_curve(rewards):
    # to do
    return


if __name__ == "__main__":
    for agent in AGENTS:
        rewards = np.load(f"results/{agent}/rewards.npy")
        plot_curve(rewards)
