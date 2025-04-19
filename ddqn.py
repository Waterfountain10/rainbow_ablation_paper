import argparse
import random
import gymnasium as gym
from typing import Dict, List
from matplotlib import pyplot as plt
import numpy as np
import sys
import os

from LoadData import load_dataset
from util.running_mean import running_mean
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dqn import DQN
import torch
import torch.nn.functional as F
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK


class DDQN(DQN):
    def __init__(
        self,
        envs: List[gym.Env],
        mem_size: int,
        batch_size: int,
        target_update_freq: int,
        epsilon_decay: float,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        alpha: float = 1e-3,
    ):
        super().__init__(
            envs=envs,
            mem_size=mem_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            epsilon_decay=epsilon_decay,
            max_epsilon=max_epsilon,
            min_epsilon=min_epsilon,
            gamma=gamma,
            alpha=alpha,
        )

    def _compute_dqn_loss(
        self, samples: Dict[str, np.ndarray]
    ) -> torch.Tensor:
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"]).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(samples["rews"]).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(samples["done"]).unsqueeze(1).to(self.device)

        curr_q_values = self.dqn_network(state)
        curr_q = curr_q_values.gather(1, action)

        with torch.no_grad():
            ############# different from dqn
            # best_action <- get best next action from Q1
            next_q_values = self.dqn_network(next_state)
            best_actions = next_q_values.argmax(dim=1, keepdim=True)

            # next_q <- Q2(best action) (get expected reward based on Q2)
            next_target_q_values = self.dqn_target(next_state)
            next_q = next_target_q_values.gather(1, best_actions)

            # compute target
            q_target = reward + self.gamma * next_q * (1 - done)
            ############# different from dqn

        loss = F.smooth_l1_loss(curr_q, q_target)

        return loss

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

if __name__ == "__main__":
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # =============== hyperparams ==================
    WINDOW_SIZE = 200
    NUMBER_STEPS = 2000

    DEFAULT_MEMORY_SIZE = 80000  # find best
    DEFAULT_BATCH_SIZE = 64  # find best

    # TODO target_update_freq needs to be different whether using ddqn or not
    DEFAULT_TARGET_UPDATE_FREQ = 32000  # find best
    TARGET_UPDATE_FREQ_DQN = 300

    DEFAULT_LEARNING_RATE = 1e-4  # find best
    DEFAULT_NUM_EPISODES = 700
    DEFAULT_EPSILON_DECAY_STEPS = (
        NUMBER_STEPS * DEFAULT_NUM_EPISODES * 0.7
    )  # want epsilon be be at minimum around 70% in the training

    DEFAULT_MIN_EPSILON = 0.01  # find best

    default_omega = 0.6
    default_beta = 0.4
    default_n_step = 3

    args = parse_args()

    MEMORY_SIZE = args.memory_size
    BATCH_SIZE = args.batch_size
    TARGET_UPDATE_FREQ = args.target_update_freq
    # TODO target_update_freq needs to be different whether using ddqn or not
    TARGET_UPDATE_FREQ_DQN = 300  # Keep or make configurable?

    EPSILON_DECAY_STEPS = int(args.epsilon_decay_steps)
    LEARNING_RATE = args.lr
    NUM_TOTAL_EPISODES = args.num_episodes
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
        "useNoisy": True,
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

    # envs = [
    #     gym.make(
    #             "forex-v0",
    #             df=FOREX_EURUSD_1H_ASK,
    #             window_size=WINDOW_SIZE,
    #             frame_bound=(WINDOW_SIZE, WINDOW_SIZE + NUMBER_STEPS + 1),
    #             unit_side="right",
    #         )
    # ]

    agent = DDQN(
        envs=envs,
        mem_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        epsilon_decay=EPSILON_DECAY_STEPS,
        alpha=LEARNING_RATE,
        min_epsilon=MIN_EPSILON,
        # agent_config=rainbow_config,
        # combined_params=default_params,
    )

    rewards = agent.train(NUM_TOTAL_EPISODES)
    # rewards = [1]
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
