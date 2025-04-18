import argparse
from typing import Dict, List, Tuple
from matplotlib.axis import Ticker
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random

import gymnasium as gym
import gym_anytrading
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL

from LoadData import load_dataset
from util.NeuralNet import NeuralNet
from util.ReplayBuffer import ReplayBuffer
import torch.nn.functional as F
from tqdm import tqdm

from numbers import Number
from util.running_mean import running_mean
import imageio  # for testing recording agents
import torch.nn.functional as F
from tqdm import tqdm

import ale_py

gym.register_envs(ale_py)

from util.NeuralNet import NeuralNet
from util.ReplayBuffer import ReplayBuffer


class DQN:
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
        """Init"""
        self.envs = envs
        self.env = random.choice(self.envs)
        self.obs_shape = self.env.observation_space.shape
        assert self.obs_shape is not None
        self.memory = ReplayBuffer(self.obs_shape, mem_size, batch_size=batch_size)
        self.gamma = gamma
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon

        # for linear
        self.epsilon_decay_rate = (
            (max_epsilon - min_epsilon) / epsilon_decay if epsilon_decay > 0 else 0
        )

        # for exponential decay rate: max * (decayRate)^eps_decay = min
        self.eps_exp_decay_rate = (min_epsilon / max_epsilon) ** (1.0 / epsilon_decay)

        # self.state_size = env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
        else:
            raise ValueError("Action space must be discrete")

        self.device = "cpu"

        # comment/uncomment below to use cpu/gpu
        # if torch.cuda.is_available():
        #     self.device = "cuda"
        # if torch.mps.is_available():
        #     self.device = "mps"

        self.dqn_network = NeuralNet(self.obs_shape, int(self.action_dim)).to(
            self.device
        )
        self.dqn_target = NeuralNet(self.obs_shape, int(self.action_dim)).to(
            self.device
        )
        # make identical copies of the neural net
        self.dqn_target.load_state_dict(self.dqn_network.state_dict())

        self.dqn_target.train(False)
        self.optimizer = torch.optim.Adam(self.dqn_network.parameters(), lr=alpha)
        self.batch_size = batch_size
        self.testing = False
        self.target_update_freq = target_update_freq
        self.total_steps = 0
        self.updating_eps = True

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            obs_flat = obs.flatten()
            obs_tensor = (
                torch.as_tensor(obs_flat, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                q_vaues = self.dqn_network(obs_tensor)
            return q_vaues.argmax().item()

    def step(
        self, state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.float32, bool]:
        """
        Go from current state -> next_state (and return everything related to this transition)
        Returns:
            action, next_state, reward, done
        """
        action = self.select_action(state)
        next_state, reward, terminated, trucated, _ = self.env.step(action)
        reward = float(reward)
        done = terminated or trucated

        self.memory.store(state, int(action), reward, next_state, done)
        self.total_steps += 1

        # linear decay
        # self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

        # exp decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.eps_exp_decay_rate)

        if self.epsilon == self.min_epsilon and self.updating_eps:
            self.updating_eps = False
            print("epsilon at minimum")

        return action, next_state, np.float32(reward), done

    def update_model(self) -> float:
        samples = self.memory.sample_batch()
        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"]).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(samples["rews"]).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(samples["done"]).unsqueeze(1).to(self.device)

        # G_t = r + gamma * v(s_{t+1})
        q_values = self.dqn_network(state)
        q_current = q_values.gather(1, action)

        with torch.no_grad():
            q_next = self.dqn_target(next_state)
            max_q_next = q_next.max(dim=1, keepdim=True)[0]
            q_target = reward + self.gamma * max_q_next * (1 - done)

        loss = F.smooth_l1_loss(
            q_current, q_target
        )  # reduction="none" for prioritized BUFfer
        # loss = F.mse_loss(q_current, q_target)

        return loss

    def _target_hard_update(self):
        """Every target_update_freq steps, target_net <- copy(current_net)"""
        self.dqn_target.load_state_dict(self.dqn_network.state_dict())

    def train(self, num_episodes, show_progress=True):
        window_size = min(10, num_episodes // 10)
        rewards = []

        episode_bar = None
        if show_progress:
            episode_bar = tqdm(total=num_episodes, desc="Episodes", leave=False)

        for episode in range(num_episodes):
            self.env = random.choice(self.envs)
            state, _ = self.env.reset()
            done = False
            ep_reward = 0
            steps_n = 0

            while not done:
                action, next_state, reward, done = self.step(state)

                # only update if batch has enough samples
                if len(self.memory) >= self.batch_size:
                    loss = self.update_model()

                if self.total_steps % self.target_update_freq == 0:
                    self._target_hard_update()

                state = next_state
                ep_reward += reward
                steps_n += 1

            # update target network if needed
            # self._target_hard_update()

            rewards.append(ep_reward)
            if show_progress and episode_bar is not None:
                # recent_rewards = (
                #     rewards[-window_size:] if len(rewards) >= window_size else rewards
                # )
                # avg_reward = sum(recent_rewards) / len(recent_rewards)

                # # Same progress bar update as before
                # postfix_dict = {
                #     "reward": f"{ep_reward:.1f}",
                #     "avg": f"{avg_reward:.1f}",
                #     "steps": steps_n,
                # }

                # postfix_dict["ε"] = f"{self.epsilon:.3f}"

                episode_bar.update(1)
                # episode_bar.set_postfix(postfix_dict)
                episode_bar.set_postfix(
                    reward=f"{ep_reward:.1f}",
                    steps=steps_n,
                    epsilon=f"{self.epsilon:.2f}",
                    rews_avg=f"{np.mean(rewards):.2f}",
                )

        if show_progress and episode_bar is not None:
            episode_bar.close()
        self.env.close()
        return rewards

    """def plot(self):
        plt.figure(figsize=10,5)
        plt.plot(rewards, label="episode Reward", alpha=0.6)

        if len(rewards) >= 10: # apply cumsum sliding mean
            smoothed = running_mean(rewards, window_size =10)
            plt.plot(
                range(10 - 1, len(rewards)), smoothed, label="smoothed window 10", linewidth=2
            )
        plt.title("DQN training rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout
        plt.show()

        # also save png
        plt.savefig("results/rewards_DQN.png")
        print("Plot saved to results/rewards.png")
        """


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

    EPSILON_DECAY_STEPS = args.epsilon_decay_steps
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
        data_set = load_dataset(set)
        envs.append(
            gym.make(
                "forex-v0",
                df=data_set,
                window_size=WINDOW_SIZE,
                frame_bound=(WINDOW_SIZE, WINDOW_SIZE + NUMBER_STEPS + 1),
                unit_side="right",
            )
        )

    agent = DQN(
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
