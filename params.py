import argparse
import random

import numpy as np
import torch

'''
===========================================================================
                    Default Hyperparameter values
===========================================================================
'''
# Data Parameters
WINDOW_SIZE = 200

# Training Parameters
NUMBER_STEPS = 700
NUMBER_TRAIN_EPISODES = 700
NUMBER_TEST_EPISODES = 200
NUM_TOTAL_EPISODES = NUMBER_TRAIN_EPISODES+NUMBER_TEST_EPISODES                # changed for atari (stocks was 500 + 200)


# General Parameters

MEMORY_SIZE = 80000                     # changed for atari (stocks was 80 000)

BATCH_SIZE = 32                         # changed for atari (stocks was 256)
LEARNING_RATE = 1e-4                    # changed for atari (stocks was 5e-4)
TARGET_UPDATE_FREQ = 8000               # changed for atari (stocks was 1000)
MIN_EPSILON = 0.01                      # changed for atari (stocks was 0.1)
EPSILON_DECAY_STEPS = (
    NUMBER_STEPS * NUMBER_TRAIN_EPISODES * 0.7
)
HIDDEN_DIM = 512                        # changed for atari (stocks was 256)

# Model Specific Parameters
OMEGA = 0.6
BETA = 0.4
NSTEP = 3
TD_EPSILON = 1e-6
V_MIN = -10.0                           # changed for atari (stocks was -10 and 10)
V_MAX = 10.0
ATOM_SIZE = 51
SIGMA_INIT = 0.5
GAMMA = 0.99

'''
===========================================================================
                      Argument Parsing for Script
===========================================================================
'''


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent.")
    parser.add_argument(
        "-memory_size", type=int, default=MEMORY_SIZE, help="Replay buffer size"
    )
    parser.add_argument(
        "-batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "-target_update_freq",
        type=int,
        default=TARGET_UPDATE_FREQ,
        help="Frequency of target network updates",
    )
    parser.add_argument(
        "-epsilon_decay_steps",
        type=float,
        default=EPSILON_DECAY_STEPS,
        help="Steps over which epsilon decays",
    )
    parser.add_argument(
        "-lr", type=float, default=LEARNING_RATE, help="Learning rate"
    )
    parser.add_argument(
        "-num_episodes",
        type=int,
        default=NUM_TOTAL_EPISODES,
        help="Number of training episodes",
    )
    parser.add_argument(
        "-min_epsilon",
        type=float,
        default=MIN_EPSILON,
        help="Minimum epsilon value",
    )
    parser.add_argument(
        "-n_step",
        type=int,
        default=NSTEP,
        help="Number of steps taken in multi-step learning"
    )
    parser.add_argument("-omega",
        type=float,
        default=OMEGA,
        help="Omega value"
        )
    parser.add_argument("-beta",
        type=float,
        default=BETA,
        help="Beta value"
        )
    parser.add_argument(
        "-td_epsilon",
        type=float,
        default=TD_EPSILON,
        help="TD epsilon value",)
    parser.add_argument(
        "-v_min",
        type=float,
        default=V_MIN,
        help="Minimum value for the value distribution",
    )
    parser.add_argument(
        "-v_max",
        type=float,
        default=V_MAX,
        help="Maximum value for the value distribution",
    )
    parser.add_argument(
        "-atom_size",
        type=int,
        default=ATOM_SIZE,
        help="Number of atoms for the value distribution",
    )
    parser.add_argument(
        "-sigma_init",
        type=float,
        default=SIGMA_INIT,
        help="Initial sigma value for Noisy layers",
    )
    parser.add_argument(
        "-gamma",
        type=float,
        default=GAMMA,
        help="Discount factor for future rewards",
    )
    parser.add_argument(
        "-useDouble",
        action="store_true",
        default=False,
        help="Enable Double DQN"
    )
    parser.add_argument(
        "-usePrioritized",
        action="store_true",
        default=False,
        help="Enable Prioritized Experience Replay",
    )
    parser.add_argument(
        "-useDuel",
        action="store_true",
        default=False,
        help="Enable Duel Network Architecture",
    )
    parser.add_argument(
        "-useNoisy",
        action="store_true",
        default=False,
        help="Enable Noisy Network Architecture",
    )
    parser.add_argument(
        "-useDistributive",
        action="store_true",
        default=False,
        help="Enable Distributive Network Architecture",
    )
    parser.add_argument(
        "-useNstep",
        action="store_true",
        default=False,
        help="Enable N-step Returns",
    )
    parser.add_argument(
        "-hidden_dim",
        type=float,
        default=HIDDEN_DIM,
        help="Minimum value for the value distribution",
    )
    parser.add_argument(
        "-env",
        type=str,
        default="",
        help="name of the environment"
    )
    parser.add_argument(
        "-ablation",
        action="store_true",
        default=False,
        help="max asked me to add this"
    )

    return parser.parse_args()
