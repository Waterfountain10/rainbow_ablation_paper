import gymnasium as gym
from typing import Dict
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dqn import DQN
from util.ReplayBuffer import ReplayBufferReturn
import torch
import torch.nn.functional as F
import ale_py


class DDQN(DQN):
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
        super().__init__(
            env=env,
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
        self, samples: Dict[str, np.ndarray] | ReplayBufferReturn
    ) -> torch.Tensor:
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"]).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(samples["rews"]).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(samples["done"]).unsqueeze(1).to(self.device)

        curr_q_values = self.dqn_network(state)
        curr_q = curr_q_values.gather(1, action)

        with torch.no_grad():
            # use online network to get actions
            next_q_values = self.dqn_network(next_state)
            best_actions = next_q_values.argmax(dim=1, keepdim=True)

            # use target network to get best action
            next_target_q_values = self.dqn_target(next_state)
            next_q = next_target_q_values.gather(1, best_actions)

            # compute target
            target = reward + self.gamma * next_q * (1 - done)

        loss = F.smooth_l1_loss(curr_q, target)

        return loss


if __name__ == "__main__":
    # Parameters for DQN
    MEMORY_SIZE = 10000
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 10000
    EPSILON_DECAY_STEPS = 100000
    LEARNING_RATE = 1e-4
    NUM_EPISODES = 100  # Small number for testing
    MIN_EPSILON = 0.05

    # env = gym.make(
    #     "forex-v0",
    #     df=FOREX_EURUSD_1H_ASK,
    #     window_size=10,
    #     frame_bound=(10, int(0.25 * len(FOREX_EURUSD_1H_ASK))),
    #     unit_side="right",
    # )

    gym.register_envs(ale_py)
    env = gym.make("ALE/Assault-ram-v5", render_mode=None, max_episode_steps=1000)

    agent = DQN(
        env=env,
        mem_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        epsilon_decay=EPSILON_DECAY_STEPS,
        alpha=LEARNING_RATE,
        min_epsilon=MIN_EPSILON
    )

    rewards = agent.train(NUM_EPISODES)
    print(np.mean(rewards))
