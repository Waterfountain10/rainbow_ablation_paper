import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from util.running_mean import running_mean
from params import NUM_TOTAL_EPISODES

def plot_all_checkpoints(directory="atari_checkpoints", smoothing_window=50, figsize=(12, 8)):
    """
    Plot all .npy reward files in the specified directory.
    Each file should contain a numpy array of shape (3, n_episodes) representing 3 trials.

    Args:
        directory (str): Directory containing .npy files with rewards
        smoothing_window (int): Window size for smoothing the reward curves
        figsize (tuple): Figure size (width, height) in inches
    """
    # Find all .npy files in directory
    npy_files = glob.glob(os.path.join(directory, "*.npy"))

    if not npy_files:
        print(f"No .npy files found in {directory}")
        return

    # Set up the plot
    plt.figure(figsize=figsize)

    # Prepare color cycle (using tab10 colormap for distinct colors)
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(npy_files))))

    # Track min/max values for y-axis scaling
    all_min_rewards = []
    all_max_rewards = []

    # Plot each file
    for i, file_path in enumerate(npy_files):
        # Extract meaningful name from filename
        filename = os.path.basename(file_path)
        name = filename.replace('.npy', '')

        try:
            # Add allow_pickle=True to load object arrays
            data = np.load(file_path, allow_pickle=True)

            avg = [float((x + y + z) / 3) for x, y, z in zip(data[0], data[1], data[2])]

            # Track min/max values for scaling
            all_min_rewards.append(min(avg))
            all_max_rewards.append(max(avg))

            smoothed_avg = running_mean(avg, smoothing_window)
            smoothed_avg = np.array(smoothed_avg)
            plot_label = f"{name[15:]} (smoothed)"
            plt.plot(smoothed_avg, label=plot_label, color=colors[i % len(colors)], alpha=0.7)

        except Exception as e:
            print(f"Error plotting {filename}: {e}")

    # Set plot labels and title
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title(f'Training Rewards Comparison for {name[:14]} (Averaged over 3 Trials)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    # Set x-axis to show integer episode numbers
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add legend with smaller font outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Set y-axis limits with some padding
    if all_min_rewards:
        y_min = float(min(all_min_rewards))
        y_max = float(max(all_max_rewards))
    else:
        y_min = 0.0
        y_max = 100.0

    padding = (y_max - y_min) * 0.1  # 10% padding
    plt.ylim(y_min - padding, y_max + padding)

    # Adjust layout to make room for the legend
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(directory, "rewards_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(directory, 'rewards_comparison.png')}")

if __name__ == "__main__":
    plot_all_checkpoints()
