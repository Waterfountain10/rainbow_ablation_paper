from typing import Dict, Tuple
from matplotlib.axis import Ticker
import numpy as np
import torch

import gymnasium as gym

# import gym_anytrading
import pandas as pd

# import ale_py
import torch.nn.functional as F
from tqdm import tqdm
from util.NeuralNet import NeuralNet
from util.ReplayBuffer import ReplayBuffer

# --- FinRL Imports ---
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.data_processor import DataProcessor
from finrl import config_tickers  # Or provide your own list
import os
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime


class DQN:
    def __init__(
        self,
        env: gym.Env,
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
        self.env = env
        # storage = LazyTensorStorage(max_size=mem_size)
        # self.buffer = TensorDictReplayBuffer(
        #     storage=storage, batch_size=batch_size
        # )  # TODO: could add prefetch (multithreaded thing)
        self.obs_shape = env.observation_space.shape
        self.memory = ReplayBuffer(self.obs_shape, mem_size, batch_size=batch_size)
        self.gamma = gamma
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_rate = (
            (max_epsilon - min_epsilon) / epsilon_decay if epsilon_decay > 0 else 0
        )
        # self.state_size = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.device = "cpu"

        # comment/uncomment below to use cpu/gpu
        # if torch.cuda.is_available():
        #     self.device = "cuda"
        # if torch.mps.is_available():
        #     self.device = "mps"

        self.dqn_network = NeuralNet(self.obs_shape, self.action_dim).to(self.device)
        self.dqn_target = NeuralNet(self.obs_shape, self.action_dim).to(self.device)
        # make identical copies of the neural net
        self.dqn_target.load_state_dict(self.dqn_network.state_dict())

        self.dqn_target.train(False)
        self.optimizer = torch.optim.Adam(self.dqn_network.parameters(), lr=alpha)
        self.batch_size = batch_size
        self.testing = False
        self.target_update_freq = target_update_freq
        self.total_steps = 0

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

    def step(self, state: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """
        Returns:
            action, next_state, reward, done
        """
        action = self.select_action(state)
        next_state, reward, terminated, trucated, _ = self.env.step(action)
        done = terminated or trucated

        self.memory.store(state, action, reward, next_state, done)
        self.total_steps += 1
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

        return action, next_state, reward, done

    def update_model(self) -> torch.TensorType:
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

        loss = F.smooth_l1_loss(q_current, q_target)
        # loss = F.mse_loss(q_current, q_target)

        return loss

    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn_network.state_dict())

    def train(self, num_episodes, show_progress=True):
        rewards = []

        if show_progress:
            episode_bar = tqdm(total=num_episodes, desc="Episodes", leave=False)

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            ep_reward = 0
            steps_n = 0

            while not done:
                action, next_state, reward, done = self.step(state)
                loss = self.update_model()
                state = next_state
                ep_reward += reward
                steps_n += 1

            rewards.append(ep_reward)
            if show_progress:
                episode_bar.update(1)
                episode_bar.set_postfix(reward=f"{ep_reward:.1f}", steps=steps_n)

        if show_progress:
            episode_bar.close()
        self.env.close()
        return rewards

    def plot(self):
        pass


if __name__ == "__main__":
    # Parameters for DQN
    MEMORY_SIZE = 10000
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 1000
    EPSILON_DECAY_STEPS = 10000
    LEARNING_RATE = 1e-4
    NUM_EPISODES = 10  # Small number for testing

    # Download data directly with yfinance for simplicity
    ticker_list = ["AAPL"]
    start_date = "2018-01-01"
    end_date = "2019-01-01"

    print("Downloading data...")
    # Direct yfinance approach to avoid processor issues
    df = yf.download(ticker_list, start=start_date, end=end_date)
    df = df.reset_index()  # Make date a column instead of index

    # Convert to expected format
    df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adjcp",
        },
        inplace=True,
    )

    # Add ticker column
    df["tic"] = "AAPL"

    # Add some basic technical indicators
    print("Adding technical indicators...")
    df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["rsi_30"] = 100 - (
        100
        / (
            1
            + (
                df["close"].diff(1).clip(lower=0).rolling(30).mean()
                / -df["close"].diff(1).clip(upper=0).rolling(30).mean()
            )
        )
    )

    # Drop NaN values that result from indicators
    df.dropna(inplace=True)

    # Reset index to ensure proper indexing
    df = df.reset_index(drop=True)

    # This is critical: Set date as the index
    df["date"] = pd.to_datetime(df["date"])

    # Set up the environment
    print("Setting up environment...")
    stock_dimension = len(df["tic"].unique())
    state_space = 1 + 2 * stock_dimension + 2  # cash + shares + 2 technical indicators

    # Initialize with zero shares for each stock
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,  # Max shares to trade
        "initial_amount": 10000,  # Initial cash
        "buy_cost_pct": 0.001,  # Transaction fee for buying
        "sell_cost_pct": 0.001,  # Transaction fee for selling
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": ["macd", "rsi_30"],
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "num_stock_shares": num_stock_shares,
        #"if_discrete": True,  # Ensure discrete action space
    }

    # Create the environment
    env = StockTradingEnv(df=df, **env_kwargs)

    # Initialize and train the DQN agent
    print("Initializing agent...")
    agent = DQN(
        env=env,
        mem_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        epsilon_decay=EPSILON_DECAY_STEPS,
        alpha=LEARNING_RATE,
    )

    print("Training agent...")
    rewards = agent.train(NUM_EPISODES)
    print(f"Average reward over {NUM_EPISODES} episodes: {np.mean(rewards)}")

    # Plot the rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("dqn_rewards.png")
    plt.show()
