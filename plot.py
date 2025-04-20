import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from util.running_mean import running_mean
from params import NUM_TOTAL_EPISODES, NUMBER_TEST_EPISODES

def plot_all_checkpoints(directory="atari_checkpoints",
                         smoothing_window=50,
                         figsize=(12, 8)):
    npy_files = glob.glob(os.path.join(directory, "*.npy"))
    if not npy_files:
        print(f"No .npy files found in {directory}")
        return

    split_idx = NUM_TOTAL_EPISODES - NUMBER_TEST_EPISODES

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(npy_files)))

    all_min, all_max = [], []
    test_stats = []

    for i, fp in enumerate(npy_files):
        label = os.path.basename(fp).replace('.npy','')[15:]
        data = np.load(fp, allow_pickle=True)
        avg = [(x + y + z)/3.0 for x,y,z in zip(data[0], data[1], data[2])]
        all_min.append(min(avg)); all_max.append(max(avg))

        sm = running_mean(avg, smoothing_window)
        ax.plot(sm, label=f"{label} (smoothed)",
                color=colors[i], alpha=0.7)

        test_stats.append((label, np.mean(avg[split_idx:]), colors[i]))

    # split line
    ax.axvline(split_idx, color='k', linestyle='--', linewidth=1)

    # y‑limits
    y_min, y_max = min(all_min), max(all_max)
    yr = y_max - y_min; pad = yr * 0.05
    ax.set_ylim(y_min - pad, y_max + pad)

    # low labels
    ax.text(0.25, -0.1, "Training",
            transform=ax.transAxes,
            ha='center', va='top', fontsize=12, alpha=0.8)
    ax.text(0.75, -0.1, "Testing\n(no training)",
            transform=ax.transAxes,
            ha='center', va='top', fontsize=12, alpha=0.8)

    # high labels at ~80% of the reward scale
    high_y = y_min + 0.8 * yr
    ax.text(split_idx/2, high_y, "Training",
            ha='center', va='center', fontsize=12, weight='bold',
            color='gray', alpha=0.6)
    ax.text(split_idx + NUMBER_TEST_EPISODES/2, high_y,
            "Testing\n(no training)",
            ha='center', va='center', fontsize=12, weight='bold',
            color='gray', alpha=0.6)

    # test‑means box, down below legend
    stats_text = "\n".join(f"{n}: {m:.1f}" for n,m,_ in test_stats)
    ax.text(1.02, 0.65, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # labels & legend
    ax.set_title('Training Rewards Comparison for RoadRunner‑ram\n'
                 '(Averaged over 3 Trials)', fontsize=14)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=10)
    plt.tight_layout()

    out = os.path.join(directory, "rewards_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out}")

if __name__ == "__main__":
    plot_all_checkpoints()
