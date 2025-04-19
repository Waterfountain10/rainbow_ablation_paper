import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from params import NUMBER_STEPS, WINDOW_SIZE, parse_args, NUMBER_TEST_EPISODES
from util.running_mean import running_mean
from LoadData import load_dataset
import gym_anytrading
from combined_agent import CombinedAgent
import os

args = parse_args()

MEMORY_SIZE = args.memory_size
BATCH_SIZE = args.batch_size
TARGET_UPDATE_FREQ = args.target_update_freq
# TODO target_update_freq needs to be different whether using ddqn or not
TARGET_UPDATE_FREQ_DQN = 300  # Keep or make configurable?

EPSILON_DECAY_STEPS = int(args.epsilon_decay_steps)
LEARNING_RATE = args.lr
NUM_EPISODES = args.num_episodes
MIN_EPSILON = args.min_epsilon

default_params = {
    "omega": args.omega,  # from 0 to 1
    "beta": args.beta,  # from 0.4 to 0.7
    "td_epsilon": 1e-6,
    "v_min": -100.0,
    "v_max": 100.0,
    "atom_size": 51,
    "n_step": args.n_step,  # find best
    "sigma_init": 0.5,
    "gamma": 0.99,
}

# ============== end of hyperparams ============


rainbow_config = {
    "useDouble": True,
    "usePrioritized": True,
    "useDuel": True,
    "useNoisy": False,
    "useNstep": True,
    "useDistributive": True,
}

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
    data_set = load_dataset(set, WINDOW_SIZE + NUMBER_STEPS + 1)
    envs.append(
        gym.make(
            "forex-v0",
            df=data_set,
            window_size=WINDOW_SIZE,
            frame_bound=(WINDOW_SIZE, len(data_set)),
            unit_side="right",
        )
    )

test_envs = []
for data_set in data_sets:
    data_set = load_dataset(set, (WINDOW_SIZE + NUMBER_STEPS + 1) + NUMBER_STEPS)
    test_envs.append(
        gym.make(
            "forex-v0",
            df=data_set,
            window_size=WINDOW_SIZE,
            frame_bound=(
                WINDOW_SIZE + NUMBER_STEPS + 1,
                len(data_set),
            ),  # logic: start at where we stopped training
            unit_side="right",
        )
    )

agent = CombinedAgent(
    envs=envs,
    test_envs=test_envs,
    mem_size=MEMORY_SIZE,
    batch_size=BATCH_SIZE,
    target_update_freq=TARGET_UPDATE_FREQ,
    epsilon_decay=EPSILON_DECAY_STEPS,
    alpha=LEARNING_RATE,
    min_epsilon=MIN_EPSILON,
    agent_config=rainbow_config,
    combined_params=default_params,
)

rewards = agent.train(NUM_EPISODES)
# print("Rewards at end:", np.mean(rewards))
# PLOT GRAPH AND SAVE IT
plt.figure(figsize=(10, 5))
plt.plot(rewards, label="episode Reward", alpha=0.6)
if len(rewards) >= 20:  # apply cumsum sliding mean
    smoothed = running_mean(rewards, window_size=20)
    plt.plot(
        range(20 - 1, len(rewards)),
        smoothed,
        label="smoothed window 20",
        linewidth=2,
    )

# Calculate and plot the average reward
average_reward = np.mean(rewards)
plt.axhline(
    average_reward,  # type: ignore
    color="r",
    linestyle="--",
    label=f"Average Reward: {average_reward:.2f}",
)

plt.title("DQN training rewards")
plt.xlabel("Episode")
plt.ylabel("Episode reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
# also save png SAVE DID NOT WORK BTW
os.makedirs("results", exist_ok=True)
plt.savefig(
    f"results/mem{MEMORY_SIZE:d}_batch{BATCH_SIZE:d}_targUpd{TARGET_UPDATE_FREQ:d}_eDecSt{EPSILON_DECAY_STEPS:d}_lr{LEARNING_RATE}_minEps{MIN_EPSILON}_omga{args.omega}_b{args.beta}_n{args.n_step:d}.png"
)
print("Plot saved to results/rewards.png")
print("rewards mean: ", np.mean(rewards))
