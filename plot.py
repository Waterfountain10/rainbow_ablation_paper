import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from util.running_mean import running_mean  # Import your custom running_mean

def plot_all_checkpoints(directory="atari_checkpoints", smoothing_window=50, figsize=(12, 8)):
    """
    Plot all .npy reward files in the specified directory.
    
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
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(npy_files)))) # type: ignore
    
    # Track min/max values for y-axis scaling
    all_min_rewards = []
    all_max_rewards = []
    
    # Plot each file
    for i, file_path in enumerate(npy_files):
        # Extract meaningful name from filename
        filename = os.path.basename(file_path)
        name = filename.replace('.npy', '')
        
        # Load the rewards
        try:
            rewards = np.load(file_path)
            
            # Track min/max for y-axis scaling
            all_min_rewards.append(np.min(rewards))
            all_max_rewards.append(np.max(rewards))
            
            # Plot raw data with low opacity
            plt.plot(rewards, alpha=0.3, color=colors[i % len(colors)])
            
            # Smooth the data using running_mean
            if len(rewards) > smoothing_window:
                # Apply running mean smoothing
                smoothed = running_mean(rewards, smoothing_window)
                
                # Plot smoothed line (account for the offset due to window size)
                x = range(smoothing_window-1, len(rewards))
                plt.plot(x, smoothed, label=name, linewidth=2, color=colors[i % len(colors)])
            else:
                # If not enough data for smoothing, just plot the raw data
                plt.plot(rewards, label=name, linewidth=2, color=colors[i % len(colors)])
                
            print(f"Plotted {filename} with {len(rewards)} episodes")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    # Set plot labels and title
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Training Rewards Comparison', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Set x-axis to show integer episode numbers
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add legend with smaller font outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Set y-axis limits with some padding
    y_min = min(all_min_rewards) if all_min_rewards else 0
    y_max = max(all_max_rewards) if all_max_rewards else 100
    padding = (y_max - y_min) * 0.1  # 10% padding
    plt.ylim(y_min - padding, y_max + padding)
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(directory, "rewards_comparison.png"), dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_all_checkpoints()
    
    # You can also specify parameters:
    # plot_all_checkpoints(directory="other_checkpoints", smoothing_window=100)
