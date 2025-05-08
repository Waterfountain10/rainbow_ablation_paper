# Slightly Worse Rainbow

[![Made with PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)  [![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors)

> **Modular Rainbow DQN re‑implementation in PyTorch with an easy‐to‑toggle component switchboard, benchmarked on three Atari games (*Seaquest*, *Asterix*, *Road Runner*) and a high‑volatility Forex trading simulator.**

# 📄 [Read our paper here!](paper/COMP579__Recreating_Rainbow___Final_Report.pdf)

Link to Highlight Video (unlisted) : https://www.youtube.com/watch?v=v_v7eYagHHs
---
## Overview
This project implements Rainbow DQN from scratch with full modularity, enabling easy toggling of individual components (Double DQN, PER, Dueling, Noisy, Distributional, N-step).

- Unified architecture: `CombinedAgent.py`, supported by modular code in `util/`
- Baseline and ablation runs organized by filename:
  - `main.py`, `max_script.py`, `script.sh` for **ablation** runs
  - `main1.py`, `max_script1.py`, `script1.sh` for **baseline** runs
- Atari game results saved in `.npy` format inside `atari_checkpoints/`
- `Legacy/` is excluded from experiments and retained for archival purposes

---
## How to Use

1. **Set up your environment:**
```bash
conda create -n rainbow python=3.10
conda activate rainbow
pip install -r requirements.txt
```

2. **SLURM users:**
- Run ablation experiments: `bash script.sh`
- Run base model experiments: `bash script1.sh`

3. **Local runs:**
- Run ablation experiments: `python max_script.py`
- Run base model experiments: `python max_script1.py`

4. **Custom configuration:**
Run `main.py` with any combination of flags from below:
```bash
python main.py -env SeaquestNoFrameskip-v4 -num_episodes 700 -useNoisy -useDuel -useDouble
```
**Available arguments include:**
- `-env`: environment name (e.g., `SeaquestNoFrameskip-v4`, `forex-v0`)
- `-num_episodes`, `-max_steps`, `-memory_size`, `-batch_size`
- `-target_update_freq`, `-epsilon_decay_steps`, `-lr`, `-gamma`
- `-n_step`, `-omega`, `-beta`, `-td_epsilon`
- `-v_min`, `-v_max`, `-atom_size`, `-sigma_init`, `-hidden_dim`
- Flags: `--useDouble`, `--usePrioritized`, `--useDuel`, `--useNoisy`, `--useDistributive`, `--useNstep`, `--ablation`

---
## Directory Structure
```text
.
├── atari_checkpoints/       # .npy reward curves from experiments (used for plot scripts)
├── agent/                   # All components for agent including combined class
├── combined_agent.py        # Main Rainbow agent logic
├── scripts                  # main scripts to run regular/ablation mode, and also bash scripts for slurm GPU jobs
├── plotting                  # Plot scripts for regular and ablation mode
...
└── Legacy/                  # Archived code, not used
```

---
## Related Papers

01. [V. Mnih et al., "Human-level control through deep reinforcement learning." Nature, 518 (7540):529–533, 2015.](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
02. [H. van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning." arXiv:1509.06461, 2015.](https://arxiv.org/pdf/1509.06461.pdf)
03. [T. Schaul et al., "Prioritized Experience Replay." arXiv:1511.05952, 2015.](https://arxiv.org/pdf/1511.05952.pdf)
04. [Z. Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning." arXiv:1511.06581, 2015.](https://arxiv.org/pdf/1511.06581.pdf)
05. [M. Fortunato et al., "Noisy Networks for Exploration." arXiv:1706.10295, 2017.](https://arxiv.org/pdf/1706.10295.pdf)
06. [M. G. Bellemare et al., "A Distributional Perspective on Reinforcement Learning." arXiv:1707.06887, 2017.](https://arxiv.org/pdf/1707.06887.pdf)
07. [R. S. Sutton, "Learning to predict by the methods of temporal differences." Machine Learning, 3(1):9–44, 1988.](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)
08. [M. Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning." arXiv:1710.02298, 2017.](https://arxiv.org/pdf/1710.02298.pdf)

---
## Contributors

Thanks goes to these wonderful people ;)

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="20%">
        <a href="https://github.com/max-fong">
          <img src="https://avatars.githubusercontent.com/u/143747815?v=4" width="100px;" alt="Max Fong"/><br />
          <sub><b>Max Fong</b></sub>
        </a><br />
      </td>
      <td align="center" valign="top" width="20%">
        <a href="https://github.com/William-Lafond">
          <img src="https://avatars.githubusercontent.com/u/98282992?v=4" width="100px;" alt="William Kiem Lafond"/><br />
          <sub><b>William Kiem Lafond</b></sub>
        </a><br />
      </td>
      <td align="center" valign="top" width="20%">
        <a href="https://github.com/denistsariov">
          <img src="https://avatars.githubusercontent.com/u/107961778?v=4" width="100px;" alt="Denis Tsariov"/><br />
          <sub><b>Denis Tsariov</b></sub>
        </a><br />
      </td>
    </tr>
  </tbody>
</table>
<!-- markdownlint-enable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

---
## Authors
* **Max Fong** – categorical module, codebase refactor, scripts.
* **William Kiem Lafond** – Prioritized, DuelNet, NoisyNet components, report writing.
* **Denis Tsariov** – base DQN/DDQN agents, N-step Learning, hyper‑parameter tuning.

Project for *COMP 579 Deep Reinforcement Learning* with Doina Precup and Isabeau Prémont-Schwarz, McGill University, 2025.


