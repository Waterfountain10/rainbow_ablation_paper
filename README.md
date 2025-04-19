# Dissecting Rainbow DQN in Quantitative Trading

[![Made with PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors)

> **Modular Rainbow DQN re‑implementation in PyTorch with an easy‐to‑toggle component switchboard, benchmarked on three Atari games (*Seaquest*, *Asterix*, *Road Runner*) and a high‑volatility Forex trading simulator.**

---
## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Directory Structure](#directory-structure)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
   * [Atari experiments](#atari-experiments)
   * [Forex experiments](#forex-experiments)
   * [Hyper‑parameter sweeps](#hyper-parameter-sweeps)
6. [Configuration](#configuration)
7. [Reproducing the Paper Results](#reproducing-the-paper-results)
8. [Logging & Checkpoints](#logging--checkpoints)
9. [Building the Report](#building-the-report)
10. [Related Papers](#related-papers)
11. [Contributors](#contributors)
12. [License](#license)
13. [Authors](#authors)

---
## Overview
Rainbow DQN \[Hessel et al., 2018\] merges six powerful extensions to the original DQN algorithm.  Our project:

* **Re‑implements Rainbow from scratch** in a **single, unified codebase** (`CombinedAgent`, `CombinedNetwork`, `CombinedBuffer`).
* **Toggles every Rainbow component** (`--useDouble`, `--usePrioritized`, `--useDuel`, `--useNoisy`, `--useDistributive`, `--useNstep`) from the command line – perfect for ablation studies.
* Benchmarks performance on **classic RL control (Atari 2600)** *and* on a **realistic financial trading task** (gym‑anytrading Forex).
* Ships with **hyper‑parameter sweep utilities**, full training logs, and LaTeX sources for the accompanying report.

---
## Key Features
| Component | Flag | Reference |
|-----------|------|-----------|
| Double DQN | `--useDouble` | Van Hasselt et al., 2016 |
| Prioritized Experience Replay | `--usePrioritized` | Schaul et al., 2015 |
| Dueling Network Architecture | `--useDuel` | Wang et al., 2016 |
| Noisy Networks | `--useNoisy` | Fortunato et al., 2017 |
| N‑step Returns | `--useNstep` | Sutton & Barto, 2018 |
| Categorical Distributional RL (C51) | `--useDistributive` | Bellemare et al., 2017 |

Combined via the **`CombinedAgent`** class, each component can be *enabled or disabled independently* for clean, reproducible ablation.

---
## Directory Structure
```text
.
├── data/                     # data utilities (e.g. currency conversion)
│   └── convert_fx.py
├── Legacy/                  # early, standalone agents kept for reference
│   ├── util/                #   └─ Buffer/Network variants
│   └── ...                  #   (ddqn.py, dqn.py, ...)
├── util/
│   ├── CombinedBuffer.py
│   ├── CombinedNetwork.py
│   ├── SegmentTree.py
│   └── running_mean.py
├── combined_agent.py         # core training logic
├── main.py                   # command‑line entry‑point
├── script.py                 # hyper‑parameter grid search helper
├── params.py                 # default CLI / training parameters
├── plot.py                   # post‑training visualisation
├── report/                   # LaTeX sources (includes `main.tex` shown above)
└── test_checkpoints/         # saved `.npy` reward curves (auto‑created)
```

> **Tip:** Every experimental flag in *`params.py`* can be overridden at runtime; see [Configuration](#configuration).

---
## Installation
> Tested on **Python 3.10+** with *PyTorch 2.1*, *gymnasium 0.29*, and *gym‑anytrading 2.0*.

```bash
# 1. Create an isolated environment
conda create -n rainbow-trading python=3.10
conda activate rainbow-trading

# 2. Install Python dependencies
pip install torch gymnasium[atari] gym-anytrading numpy matplotlib tqdm

# (Optional) If you need Atari ROMs via AutoROM:
pip install autorom[accept-rom-license]
autorom --accept-license
```

---
## Quick Start
### Atari experiments
Run full Rainbow on **Seaquest** for 700 episodes × 700 steps:
```bash
python main.py \
  -env SeaquestNoFrameskip-v4 \
  --num_episodes 700 --max_steps 700 \
  --memory_size 500000 --batch_size 32 \
  --target_update_freq 8000 --epsilon_decay_steps 490000 \
  --lr 1e-4 --omega 0.6 --beta 0.4 --gamma 0.99 \
  --sigma_init 0.5 --n_step 3 --atom_size 51 \
  --useDouble --usePrioritized --useDuel \
  --useNoisy --useDistributive --useNstep
```
(The same command works for **AsterixNoFrameskip-v4** and **RoadRunnerNoFrameskip-v4** – just change `-env`.)

### Forex experiments
```bash
python main.py \
  -env forex-v0 \                    # custom gym‑anytrading env
  --num_episodes 900 --max_steps 700 \
  --batch_size 256 --memory_size 80000 \
  --useDouble --usePrioritized --useDuel \
  --useDistributive --useNstep        # Noisy disabled (paper insight)
```

### Hyper‑parameter sweeps
`script.py` launches an **8‑way parallel grid search** (edit `num_processes` as needed):
```bash
python script.py            # runs >30 experiment combinations
```
Results land in `results/`.

---
## Configuration
All defaults live in **`params.py`**. Every field is exposed as a CLI flag, e.g.:
```bash
# halve the target update period and bump hidden size
python main.py -env AsterixNoFrameskip-v4 -target_update_freq 4000 -hidden_dim 1024
```

Key parameters:
| Variable | Default | Meaning |
|----------|---------|---------|
| `NUMBER_STEPS` | 700 | max env steps per episode |
| `NUM_TOTAL_EPISODES` | 900 | training episodes (Atari) |
| `MEMORY_SIZE` | 500 000 | replay buffer capacity |
| `LEARNING_RATE` | 1 × 10⁻⁴ | Adam/RMSprop `lr` |
| `SIGMA_INIT` | 0.5 | initial std‑dev for Noisy layers |

---
## Reproducing the Paper Results
| Experiment | Command | Expected wall‑clock |
|------------|---------|---------------------|
| **Full Rainbow – Seaquest** | see *Atari* quick‑start above | ~90 min on RTX‑3070 |
| **Ablation (No Noisy)** | add `--useNoisy` _omitted_ | –6 % training time |
| **Forex baseline DQN** | omit all `--use*` flags | ~25 min on Apple M2 |

Pre‑trained reward curves (`.npy`) live in `test_checkpoints/` and can be plotted via:
```bash
python plot.py --input test_checkpoints/FullRainbow.npy
```

---
## Logging & Checkpoints
* **Rewards** – NumPy arrays saved automatically to `test_checkpoints/<config>.npy`.

---
## Building the Report
LaTeX sources live in `report/`

```bash
cd report/
latexmk -pdf main.tex   # rebuilds Dissecting_Rainbow_DQN.pdf
```

---
## Citation
If you use this codebase, please cite:
```bibtex
@misc{lafond2025rainbow,
  title        = {Dissecting Rainbow DQN in Quantitative Trading},
  author       = {Fong, Max and Lafond, William K. and Tsariov, Denis},
  year         = {2025},
  note         = {\url{https://github.com/<your‑repo‑url>}},
}
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
## License
This repository is released under the **MIT License**.  See [`LICENSE`](LICENSE) for details.

---
## Authors
* **Max Fong** – categorical module, codebase refactor, scripts.
* **William Kiem Lafond** – Prioritized, DuelNet, NoisyNet components, report writing.
* **Denis Tsariov** – base DQN/DDQN agents, N-step Learning, hyper‑parameter tuning.

Project for *COMP 579 Deep Reinforcement Learning* with Doina Precup and Isabeau Prémont-Schwarz, McGill University, 2025.

