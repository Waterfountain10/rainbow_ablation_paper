import numpy as np
import random
import torch
import gymnasium as gym
from dqn import DQN
import matplotlib.pyplot as plt
from multistep_dqn import MultiStepDQN
from util.running_mean import running_mean

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
MEMORY_SIZE = 500  # 500
BATCH_SIZE = 32  # 32
TARGET_UPDATE_FREQ = 300
EPSILON_DECAY_STEPS = 2e4  # 2e4
LEARNING_RATE = 1e-3
NUM_EPISODES = 1000  # Small number for testing
MIN_EPSILON = 0.01
# env = gym.make(
#     "forex-v0",
#     df=FOREX_EURUSD_1H_ASK,
#     window_size=10,
#     frame_bound=(10, int(0.25 * len(FOREX_EURUSD_1H_ASK))),
#     unit_side="right",
# )
# gym.register_envs(ale_py)
# env = gym.make("ALE/Assault-ram-v5", render_mode=None, max_episode_steps=1000)
agent = DQN(
    env=env,
    mem_size=MEMORY_SIZE,
    batch_size=BATCH_SIZE,
    target_update_freq=TARGET_UPDATE_FREQ,
    epsilon_decay=EPSILON_DECAY_STEPS,
    alpha=LEARNING_RATE,
    min_epsilon=MIN_EPSILON,
)

agent = MultiStepDQN(
    env=env,
    mem_size=MEMORY_SIZE,
    batch_size=BATCH_SIZE,
    target_update_freq=TARGET_UPDATE_FREQ,
    epsilon_decay=EPSILON_DECAY_STEPS,
    alpha=LEARNING_RATE,
    min_epsilon=MIN_EPSILON,
    n_step=3,
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
