from typing import Dict, Tuple
from matplotlib.axis import Ticker
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random

import gymnasium as gym
import gym_anytrading
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL

from util.NoisyNet import NoisyNet
from util.ReplayBuffer import ReplayBuffer
import torch.nn.functional as F
from tqdm import tqdm
from util.running_mean import running_mean
from dqn import DQN



class NoisyDQN(DQN):
    '''
    In NoisyNet paper, authors usse DuelNet + DDQN, + PER, but we will only focus on DQN.

    main difference with our dqn.py is that all epsilon greedy related code is removed.

    '''
    def __init__(
        self,
        env: gym.Env,
        mem_size: int,
        batch_size: int,
        target_update_freq: int,
        #epsilon_decay: float,
        #max_epsilon: float = 1.0,
        #min_epsilon: float = 0.1,
        gamma: float = 0.99,
        alpha: float = 1e-3,
        sigma_init: float = 0.5,

    ):
        ''' Same as DQN, except i commented out all epsilon-greedy related lines'''
        self.env = env
        self.obs_shape = env.observation_space.shape
        assert(self.obs_shape is not None)
        self.memory = ReplayBuffer(self.obs_shape, mem_size, batch_size=batch_size)
        self.gamma = gamma
        #self.epsilon = max_epsilon
        #self.epsilon_decay = epsilon_decay
        #self.max_epsilon = max_epsilon
        #self.min_epsilon = min_epsilon
        #self.epsilon_decay_rate = (
        #    (max_epsilon - min_epsilon) / epsilon_decay if epsilon_decay > 0 else 0
        #)
        # self.state_size = env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = env.action_space.n
        else:
            raise ValueError("Action space must be discrete")
        self.device = "cpu"

        # override NeuralNet with NoisyNet SPECIFIC STUFF
        assert(self.obs_shape is not None)
        self.dqn_network = NoisyNet(self.obs_shape, int(self.action_dim), sigma_init=sigma_init).to(self.device)
        self.dqn_target = NoisyNet(self.obs_shape, int(self.action_dim), sigma_init=sigma_init).to(self.device)
        self.dqn_target.load_state_dict(self.dqn_network.state_dict())
        self.dqn_target.train(False)
        self.optimizer = torch.optim.Adam(self.dqn_network.parameters(), lr=alpha)

        self.batch_size = batch_size
        self.testing = False
        self.target_update_freq = target_update_freq
        self.total_steps = 0


    def select_action(self, obs: np.ndarray) -> np.ndarray:
        '''Same as in pure DQN, but no epsilon-greedy. Instead always greedy (argmax) with respect to noisy Q. '''
        #if np.random.random() < self.epsilon:
        #    return self.env.action_space.sample()
        #else:
        obs_flat = obs.flatten()
        obs_tensor = (
            torch.as_tensor(obs_flat, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            q_vaues = self.dqn_network(obs_tensor)
        return q_vaues.argmax().item()

    def step(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.float32, bool]:
        ''' Same as in pure DQN, but no epsilon decay'''
        action = self.select_action(state)
        next_state, reward, terminated, trucated, _ = self.env.step(action)
        reward = float(reward)
        done = terminated or trucated

        self.memory.store(state, int(action), reward, next_state, done)
        self.total_steps += 1
        #self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

        return action, next_state, np.float32(reward), done


    def update_model(self) -> float:
        ''' Same as in pure DQN, but we create_epsilon (not greedy, but epsilon as in noise) to refresh noise'''
        samples = self.memory.sample_batch()
        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # NOISY DQN SPECIFIC HERE!
        # refresh epsilons in noisy layers inside our NoisyNet
        self.dqn_network.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        '''Same thing as pure dqn, but loss is calculated without reduction since we want to perform a torch.mean(widht * loss) later.'''
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

        loss = F.smooth_l1_loss(q_current, q_target) # reduction="none" for PER only
        # loss = F.mse_loss(q_current, q_target)

        return loss

    def _target_hard_update(self):
        '''Same as DQN'''
        super()._target_hard_update()

    def train(self, num_episodes, show_progress=True):
        '''Same as DQN, but no epsilon'''
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

                # only update if batch has enough samples
                if len(self.memory) >= self.batch_size:
                    loss = self.update_model()

                state = next_state
                ep_reward += reward
                steps_n += 1

            # update target network if needed
            if episode % self.target_update_freq == 0:
                self._target_hard_update()

            rewards.append(ep_reward)
            if show_progress:
                episode_bar.update(1)
                episode_bar.set_postfix(reward=f"{ep_reward:.1f}", steps=steps_n)

        if show_progress:
            episode_bar.close()
        self.env.close()
        return rewards

    '''def plot(self):
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
        '''

if __name__ == "__main__":
    # Parameters for DQN
    MEMORY_SIZE = 20000
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 100
    EPSILON_DECAY_STEPS = 1500
    LEARNING_RATE = 5e-4
    NUM_EPISODES = 2000  # Small number for testing (increased it to compare with PER - will)
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    SIGMA_INIT = 0.5 # <---- NEW HYPERPARAM FOR NOISY NET


    env = gym.make("CartPole-v1")
    #env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=50)

    agent = NoisyDQN(
        env=env,
        mem_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        #epsilon_decay=EPSILON_DECAY_STEPS,
        alpha=LEARNING_RATE,
        sigma_init = SIGMA_INIT

    )

    rewards = agent.train(NUM_EPISODES)
    #print("Rewards at end:", np.mean(rewards))

    # PLOT GRAPH AND SAVE IT
    plt.figure(figsize=(10,5))
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
    plt.tight_layout()
    #plt.show()

    # also save png SAVE DID NOT WORK BTW
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/rewards_NoisyDQN.png")
    print("Plot saved to results/rewards_NoisyDQN.png")

    plt.show()
