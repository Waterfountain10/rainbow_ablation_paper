import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from util.running_mean import running_mean
from params import NUM_TOTAL_EPISODES, NUMBER_TEST_EPISODES

def _plot_group(npy_files, directory, suffix,
                smoothing_window=50, figsize=(6, 6.5)):
    base0 = os.path.basename(npy_files[0]).replace('.npy','')
    game = base0.split('_')[0].split('-')[0]

    split_idx = NUM_TOTAL_EPISODES - NUMBER_TEST_EPISODES
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0,1,len(npy_files)))

    all_min, all_max = [], []

    for i, fp in enumerate(npy_files):
        name = os.path.basename(fp).replace('.npy','')
        variant = name[len(game)+7:].lstrip('_')
        data = np.load(fp, allow_pickle=True)
        avg  = [(x+y+z)/3.0 for x,y,z in zip(data[0], data[1], data[2])]
        all_min.append(min(avg)); all_max.append(max(avg))

        sm = running_mean(avg, smoothing_window)
        ax.plot(sm, label=f"{variant}",
                color=colors[i], linewidth=1.5,
                marker='o', markersize=3, markevery=60, alpha=0.8)

    # Vertical dashed line for train/test split
    ax.axvline(split_idx, color='k', linestyle='--', linewidth=1)

    # Y-axis limits with padding
    y_min, y_max = min(all_min), max(all_max)
    yr = y_max - y_min; pad = yr * 0.05
    ax.set_ylim(y_min - pad, (y_max + pad)*0.47)

    # Title and axes
    ax.set_title(f"{game}: Agent Comparison", fontsize=11)
    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Reward', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Horizontal legend layout
    ax.legend(loc='upper left', fontsize=9, frameon=False)



    # Tight layout for compact display
    plt.tight_layout()

    # Save
    out = os.path.join(directory, f"{game}_{suffix}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out}")

def plot_all_checkpoints(directory="atari_checkpoints",
                         smoothing_window=50,
                         figsize=(6, 6.5)):
    npy_files = glob.glob(os.path.join(directory, "*.npy"))
    if not npy_files:
        print(f"No .npy files found in {directory}")
        return

    # split on whether basename contains "_-"
    ablation = [f for f in npy_files
                if "_-" in os.path.basename(f)]
    agent_cmp = [f for f in npy_files
                 if "_" in os.path.basename(f)
                 and "_-" not in os.path.basename(f)]

    if ablation:
        _plot_group(ablation, directory, "ablation")
    if agent_cmp:
        _plot_group(agent_cmp, directory, "agent_comparison")

if __name__ == "__main__":
    plot_all_checkpoints()
