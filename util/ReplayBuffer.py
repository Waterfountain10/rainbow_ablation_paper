from typing import Dict
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size: int, obs_dim: int, batch_size: int = 32):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.buffer_shape = (
            max_size,
            obs_dim,
        )  # used to be (max_len, 2 * obs_dim + 3): state, next_state, action, reward, done
        self.obs_buffer = np.zeros(self.buffer_shape, dtype=np.float32)
        self.next_obs_buffer = np.zeros(self.buffer_shape, dtype=np.float32)
        self.actions_buffer = np.zeros([max_size], dtype=np.float32)
        self.rewards_buffer = np.zeros([max_size], dtype=np.float32)
        self.done_buffer = np.zeros([max_size], dtype=np.float32)
        self.batch_size = batch_size
        self.cur_ind = 0
        self.size = 0
        self.is_full = False

    def add_entry(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ):
        """Adds an experience tuple to the buffer."""
        self.obs_buffer[self.cur_ind] = state
        self.next_obs_buffer[self.cur_ind] = next_state
        self.actions_buffer[self.cur_ind] = action
        self.rewards_buffer[self.cur_ind] = reward
        self.done_buffer[self.cur_ind] = done
        # self.buffer[self.cur_ind] = np.hstack(
        #     (state, next_state, [action, reward, done])
        # )
        self.cur_ind = (self.cur_ind + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.is_full |= self.cur_ind == 0  # Efficiently check for full buffer

    def sample(self) -> Dict[str, np.ndarray]:
        """Randomly samples a batch of experiences."""
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buffer[idxs],
            next_obs=self.next_obs_buffer[idxs],
            actions=self.actions_buffer[idxs],
            rewards=self.rewards_buffer[idxs],
            done=self.done_buffer[idxs],
        )

    def __len__(self) -> int:
        return self.size
