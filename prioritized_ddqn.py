from typing import Dict, Tuple
from matplotlib.axis import Ticker
import matplotlib.pyplot as plt
import numpy as np
import torch

import gymnasium as gym
import gym_anytrading
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL

from util.NeuralNet import NeuralNet
from util.PrioritizedBuffer import PrioritizedReplayBuffer
import torch.nn.functional as F
from tqdm import tqdm
from plot import running_mean
from dqn import DQN



class PrioritizedDQN(DQN):
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
        # PER specific parameters:
        omega : float = 0.6, # priority importance parameter
        beta : float = 0.4,  # then gets increased more later
        td_epsilon: float = 1e-6
    ):
        super().__init__(env,mem_size,batch_size,target_update_freq,epsilon_decay,max_epsilon,min_epsilon,gamma, alpha)

        # override buffer (or memory in this case)
        self.omega = omega
        self.beta = beta
        self.td_epsilon = td_epsilon
        self.obs_shape = env.observation_space.shape
        self.memory = PrioritizedReplayBuffer(self.obs_shape, mem_size, batch_size, omega, beta, td_epsilon)

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        '''Same select_action() as pure DQN, using epsilon-greedy'''
        return super().select_action(obs)

    def step(self, state: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        ''' Same as in pure DQN, but implicitly, memory stores differently under the hood'''
        return super().step(state)

    def update_model(self) -> torch.TensorType:
        '''
        Different from pure DQN, sampling buffer outputs weights and idxs,
        and loss is now calculated with weights. Also, need to update priorities, once
        '''
        samples = self.memory.sample_batch() # sample = dict with many keys
        losses = self._compute_dqn_loss(samples) # torch.Tensor

        # calculate weighted loss rather than simple loss
        weights = torch.FloatTensor(samples["weights"].reshape(-1,1)).to(self.device)
        weighted_loss = torch.mean(losses * weights)

        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        # update priorities with indeces array
        idxs = samples["idxs"] # simple array type
        td_tensor = losses.detach().cpu().numpy() # untrack the gradients since this is not used for loss calculation but just priority tracking
        td_tensor = td_tensor.squeeze()
        new_priorities = abs(td_tensor + self.td_epsilon) # p_i = |delta_i| + epsilon
        self.memory.update_priorities(idxs, new_priorities) # updates in buffer with : p_i ^ omega

        return weighted_loss.item()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        return super()._compute_dqn_loss(samples)

    def _target_hard_update(self):
        super()._target_hard_update()

    def train(self, num_episodes, show_progress=True):
        '''Same as pure DQN, but beta improves slightly after incrementing reward'''
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

                # PER specific: annealed_beta := beta + x*beta where x is either %_steps_currenlty or 100%
                fraction = min(self.total_steps / num_episodes, 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)

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
    MEMORY_SIZE = 10000
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 10
    EPSILON_DECAY_STEPS = 700
    LEARNING_RATE = 1e-4
    NUM_EPISODES = 300  # Small number for testing

    env = gym.make(
        "forex-v0",
        df=FOREX_EURUSD_1H_ASK,
        window_size=10,
        frame_bound=(10, int(0.25 * len(FOREX_EURUSD_1H_ASK))),
        unit_side="right",
    )

    agent = DQN(
        env=env,
        mem_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        epsilon_decay=EPSILON_DECAY_STEPS,
        alpha=LEARNING_RATE,
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
    plt.tight_layout
    plt.show()

    # also save png SAVE DID NOT WORK BTW
    plt.savefig("results/rewards_DQN.png")
    print("Plot saved to results/rewards.png")
