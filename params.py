import argparse
import random

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent.")
    parser.add_argument(
        "-memory_size", type=int, default=DEFAULT_MEMORY_SIZE, help="Replay buffer size"
    )
    parser.add_argument(
        "-batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "-target_update_freq",
        type=int,
        default=DEFAULT_TARGET_UPDATE_FREQ,
        help="Frequency of target network updates",
    )
    parser.add_argument(
        "-epsilon_decay_steps",
        type=float,
        default=DEFAULT_EPSILON_DECAY_STEPS,
        help="Steps over which epsilon decays",
    )
    parser.add_argument(
        "-lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate"
    )
    parser.add_argument(
        "-num_episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Number of training episodes",
    )
    parser.add_argument(
        "-min_epsilon",
        type=float,
        default=DEFAULT_MIN_EPSILON,
        help="Minimum epsilon value",
    )
    parser.add_argument(
        "-n_step", type=int, default=default_n_step, help="Multi-step return N"
    )
    parser.add_argument("-omega", type=float, default=default_omega, help="Omega value")
    parser.add_argument("-beta", type=float, default=default_beta, help="Beta value")

    return parser.parse_args()


SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# =============== hyperparams ==================
WINDOW_SIZE = 200
NUMBER_STEPS = 500
NUMBER_TEST_EPISODES = 100

DEFAULT_MEMORY_SIZE = 80000  # 80K is good
DEFAULT_BATCH_SIZE = 256  # find best

# TODO target_update_freq needs to be different whether using ddqn or not
DEFAULT_TARGET_UPDATE_FREQ = 32000  # find best
TARGET_UPDATE_FREQ_DQN = 300

DEFAULT_LEARNING_RATE = 5e-4  # find best
DEFAULT_NUM_EPISODES = 400
DEFAULT_EPSILON_DECAY_STEPS = (
    NUMBER_STEPS * DEFAULT_NUM_EPISODES * 0.7
)  # want epsilon be be at minimum around 70% in the training

DEFAULT_MIN_EPSILON = 0.10  # find best

default_omega = 0.6
default_beta = 0.4
default_n_step = 3