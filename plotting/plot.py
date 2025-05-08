import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from util.running_mean import running_mean
from params import NUM_TOTAL_EPISODES, NUMBER_TEST_EPISODES

def _plot_group(npy_files, directory, suffix,
                smoothing_window=50, figsize=(12,8),
                exclude_variants=None, min_avg=None, max_avg=None):
    # derive game name from the first file
    base0 = os.path.basename(npy_files[0]).replace('.npy','')
    game = base0.split('_')[0].split('-')[0]

    split_idx = NUM_TOTAL_EPISODES - NUMBER_TEST_EPISODES
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0,1,len(npy_files)))

    all_min, all_max = [], []
    test_stats = []

    for i, fp in enumerate(npy_files):
        name = os.path.basename(fp).replace('.npy','')
        # strip off game + separator to get the variant label

        variant = name[len(game)+7:].lstrip('_')
        data = np.load(fp, allow_pickle=True)
        if data.ndim == 1:
            avg = data[0]
        else:
            avg = [(x+y+z)/3.0 for x,y,z in zip(data[0], data[1], data[2])]

        # make sure there are test episodes to average
        if len(avg) < split_idx:
            print(f"Skipping {variant}: only {len(avg)} episodes (< split_idx {split_idx})")
            continue

        # now slice out the test part and compute its mean
        test_slice = avg[split_idx:]
        mean_val = np.mean(test_slice)

        # skip by name
        if exclude_variants and variant in exclude_variants:
            continue
        # skip if below/above thresholds (now using test‐mean)
        if min_avg is not None and mean_val < min_avg:
            continue
        if max_avg is not None and mean_val > max_avg:
            continue

        all_min.append(min(avg)); all_max.append(max(avg))

        sm = running_mean(avg, smoothing_window)
        ax.plot(sm, label=f"{variant} (smoothed)",
                color=colors[i], alpha=0.7)

        test_stats.append((variant, mean_val))

    # if no runs survived the filters, bail out
    if not all_min:
        print(f"No variants to plot for {game} after applying filters.")
        return

    # vertical line
    ax.axvline(split_idx, color='k', linestyle='--', linewidth=1)

    # y‑limits
    y_min, y_max = min(all_min), max(all_max)
    yr = y_max - y_min; pad = yr * 0.05
    ax.set_ylim(y_min - pad, y_max + pad)

    # labels below
    ax.text(0.25, -0.1, "Training",
            transform=ax.transAxes, ha='center', va='top', fontsize=12)
    ax.text(0.75, -0.1, "Testing\n(no training)",
            transform=ax.transAxes, ha='center', va='top', fontsize=12)

    # labels up high (~80%)
    high_y = y_min + 0.8 * yr
    ax.text(split_idx/2, high_y, "Training",
            ha='center', va='center', fontsize=12,
            color='gray', alpha=0.6)
    ax.text(split_idx + NUMBER_TEST_EPISODES/2, high_y,
            "Testing\n(no training)",
            ha='center', va='center', fontsize=12,
            color='gray', alpha=0.6)

    # test means box (lowered so it sits below the legend)
    stats_text = "\n".join(f"{v}: {m:.1f}" for v,m in test_stats)
    ax.text(1.02, 0.60, stats_text,
            transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # title & axes
    nice_suffix = suffix.replace('_', ' ').title()
    ax.set_title(f"Training Rewards Comparison for {game} ({nice_suffix})",
                 fontsize=14)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=10)
    plt.tight_layout()

    out = os.path.join(directory, f"{game}_{suffix}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out}")

def plot_all_checkpoints(directory="atari_checkpoints",
                         smoothing_window=50,
                         figsize=(12,8)):
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
