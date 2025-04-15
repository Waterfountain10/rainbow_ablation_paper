from collections import deque
from typing import Dict, Tuple
from util.ReplayBuffer import ReplayBuffer
import numpy as np


class NstepBuffer(ReplayBuffer):
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        size: int,
        n_step: int,
        gamma: float,
        batch_size: int = 32,
    ):
        super().__init__(obs_shape=obs_shape, size=size, batch_size=batch_size)

        ######### different from ReplayBuffer
        self.n_step_buf = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma
        ######### different from ReplayBuffer

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> bool:
        ############### different from ReplayBuffer
        # add entry n_step buffer
        buf_entry = (state, action, reward, next_state, done)
        self.n_step_buf.append(buf_entry)

        if len(self.n_step_buf) < self.n_step:
            # not enough entries yet to calculate reward for the oldest element in the queue
            return False

        # calculate reward for oldest element in queue
        n_step_reward = 0.0
        discount = 1
        # setup default final state, done (i.e. of the state of the element just added)
        final_next_state = self.n_step_buf[-1][3]
        final_done = self.n_step_buf[-1][4]

        # calculate reward for oldest element in queue by accumulating rewards and gamma forward
        for i in range(self.n_step):
            s, a, r, ns, d = self.n_step_buf[i]
            n_step_reward += discount * r
            discount *= self.gamma
            if d:
                # to avoid training on next episode, need to check if episode of the oldest element in the queue is over
                # in case it is, need to break
                final_next_state = ns
                final_done = True
                break

        oldest_state, oldest_action, oldest_reward, oldest_next_state, oldest_done = (
            self.n_step_buf[0]
        )

        # once we have accumulated the forward reward for the oldest element in the queue, we can finally add it to the ReplayBuffer
        self.state_buf[self.curr_ind] = oldest_state
        self.next_state_buf[self.curr_ind] = final_next_state
        self.acts_buf[self.curr_ind] = oldest_action
        self.rewards_buf[self.curr_ind] = n_step_reward
        self.done_buf[self.curr_ind] = final_done
        ################## different from ReplayBuffer

        self.curr_ind = (
            (self.curr_ind + 1) % self.max_size
        )  # if at end, go back to start and replace the oldest experiences
        self.size = min(
            self.size + 1, self.max_size
        )  # buffer size increase (capped at max_size) -> replace oldest back in start

        ##################################### also different :) (need to return boolean)
        return True
