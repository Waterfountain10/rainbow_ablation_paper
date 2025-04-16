import numpy as np
import random
import torch
import gymnasium as gym
from dqn import DQN
import matplotlib.pyplot as plt
from multistep_dqn import MultiStepDQN
from ddqn import DDQN
from util.running_mean import running_mean
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK
from LoadData import load_dataset
import gym_anytrading

# Parameters for DQN
# MEMORY_SIZE = 20000
# BATCH_SIZE = 64
# TARGET_UPDATE_FREQ = 100
# EPSILON_DECAY_STEPS = 1500  # used to be 1500
# LEARNING_RATE = 5e-4
# NUM_EPISODES = (
#     2000  # Small number for testing (increased it to compare with PER - will)
# )
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
env = gym.make("CartPole-v1")
# * Parameters I (denis) was using and found to produce better results
MEMORY_SIZE = 100000
BATCH_SIZE = 64  # 32
TARGET_UPDATE_FREQ = 32000
EPSILON_DECAY_STEPS = 1e6  # 2e4
LEARNING_RATE = 6.25e-5
NUM_EPISODES = 1000  # Small number for testing
MIN_EPSILON = 0.01

data_sets = [
    "data/AUDUSD_H4.csv",
    "data/CADUSD_H4.csv",
    "data/CHFUSD_H4.csv",
    "data/EURUSD_H4.csv",
    "data/GBPUSD_H4.csv",
    "data/NZDUSD_H4.csv",
]

envs = []
for set in data_sets:
    data_set = load_dataset(set)
    envs.append(
        gym.make(
            "forex-v0",
            df=data_set,
            window_size=10,
            frame_bound=(10, int(0.25 * len(data_set))),
            unit_side="right",
        )
    )

# env = gym.make(
#     "forex-v0",
#     df=DATA_SET,
#     window_size=10,
#     frame_bound=(10, int(0.25 * len(DATA_SET))),
#     unit_side="right",
# )
# print(0.25 * len(DATA_SET))
# gym.register_envs(ale_py)
# env = gym.make("ALE/Assault-ram-v5", render_mode=None, max_episode_steps=1000)
# agent = DQN(
#     env=env,
#     mem_size=MEMORY_SIZE,
#     batch_size=BATCH_SIZE,
#     target_update_freq=TARGET_UPDATE_FREQ,
#     epsilon_decay=EPSILON_DECAY_STEPS,
#     alpha=LEARNING_RATE,
#     min_epsilon=MIN_EPSILON,
# )

agent = DDQN(
    env=envs[0],
    mem_size=MEMORY_SIZE,
    batch_size=BATCH_SIZE,
    target_update_freq=TARGET_UPDATE_FREQ,
    epsilon_decay=EPSILON_DECAY_STEPS,
    alpha=LEARNING_RATE,
    min_epsilon=MIN_EPSILON,
)
rewards = agent.train(NUM_EPISODES)
# print("Rewards at end:", np.mean(rewards))
# PLOT GRAPH AND SAVE IT
plt.figure(figsize=(10, 5))
plt.plot(rewards, label="episode Reward", alpha=0.6)
if len(rewards) >= 10:  # apply cumsum sliding mean
    smoothed = running_mean(rewards, window_size=10)
    plt.plot(
        range(10 - 1, len(rewards)),
        smoothed,
        label="smoothed window 10",
        linewidth=2,
    )
plt.title("DQN training rewards")
plt.xlabel("Episode")
plt.ylabel("Episode reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# also save png SAVE DID NOT WORK BTW
# os.makedirs("results", exist_ok=True)
# plt.savefig("results/rewards_DQN.png")
# print("Plot saved to results/rewards.png")
plt.show()
print("rewards mean: ", np.mean(rewards))
