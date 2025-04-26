# PS: in the Rainbow paper, they didn't plot this as a standalone multi-step learning agent, but only used this "ingredient" inside the cumulative Rainbow algorithm
# I think it can still be interesting to isolate multistep learning as a standalone agent like this : DQN + N-step learning agent!!

# to do by denis

from typing import Tuple
from numpy import float32, ndarray
from dqn import DQN
import gymnasium as gym
from util.NstepBuffer import NstepBuffer
import numpy as np


class MultiStepDQN(DQN):
    def __init__(
        self,
        env: gym.Env,
        mem_size: int,
        batch_size: int,
        target_update_freq: int,
        epsilon_decay: float,
        n_step: int,
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
        self.n_step = n_step
        assert self.obs_shape is not None
        self.memory_n = NstepBuffer(
            self.obs_shape, mem_size, batch_size=batch_size, n_step=n_step, gamma=gamma
        )

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
        self.total_steps += 1

        ################## different from dqn
        have_enough_in_buffer = self.memory_n.store(
            state=state,
            action=int(action),
            reward=reward,
            next_state=next_state,
            done=done,
        )
        if have_enough_in_buffer:
            # need to add here bc in train it uses len(self.memory) and don't want to override train()
            self.memory.store(state, int(action), reward, next_state, done)
        ################## different from dqn

        # linear decay
        # self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

        # exp decay
        self.epsilon = max(self.min_epsilon, self.epsilon * self.eps_exp_decay_rate)

        if self.epsilon == self.min_epsilon and self.updating_eps:
            self.updating_eps = False
            print("epsilon at minimum")

        return action, next_state, np.float32(reward), done

    def update_model(self) -> float:
        samples = self.memory_n.sample_batch()

        ##################### different from dqn
        # normal_loss = self._compute_dqn_loss(samples)

        # need to calculate loss on gamma^{self.n_step} bc we accumulated gamma in the forward look
        gamma_temp = self.gamma
        self.gamma = self.gamma**self.n_step
        loss = self._compute_dqn_loss(
            samples=samples
        )  # TODO: if bad performance due to variance, can combine n-step loss with normal loss (i.e. uncomment the normal loss line and add it to loss)
        # set gamma back to what it is supposed to be
        self.gamma = gamma_temp
        ###################### different from dqn

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
