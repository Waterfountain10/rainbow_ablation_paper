from collections import deque
from typing import Dict, Tuple, TypedDict, cast
import numpy as np
from util.SegmentTree import MinTree, SumTree


class CombinedBuffer:
    def __init__(self, obs_shape: Tuple[int, ...], size: int, batch_size: int = 32, buffer_config={
        "usePrioritized": False,
        "useNstep": False,
    },
        combined_params={
        # Nstep params
        "n_step": 1,
        "gamma": 0.99,

        # Prioritized params
        "omega": 0.6,
        "beta": 0.6,
        "td_epsilon": 1e-6
    },
    ):

        self.buffer_config = buffer_config
        obs_buffer_shape = [size] + list(obs_shape)
        self.obs_dim = obs_shape
        self.state_buf = np.zeros(obs_buffer_shape, dtype=np.float32)
        self.next_state_buf = np.zeros(obs_buffer_shape, dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.int32)
        self.rewards_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size)
        self.max_size = size
        self.batch_size = batch_size
        self.curr_ind = 0
        self.size = 0

        # Nstep specific fields
        if self.buffer_config["useNstep"]:
            self.n_step_buf = deque(maxlen=combined_params["n_step"])
            self.n_step = combined_params["n_step"]
            self.gamma = combined_params["gamma"]

        # Prioritized specific fields
        if self.buffer_config["usePrioritized"]:
            # Create 2 trees to track sum_priorities, and minimum_priority:
            tree_capacity = 1
            while tree_capacity < self.max_size:
                tree_capacity *= 2
            self.sum_priority_tree = SumTree(tree_capacity)
            self.min_priority_tree = MinTree(tree_capacity)
            self.max_priority = 1.0
            self.tree_pointer = 0
            self.omega = combined_params["omega"]
            self.beta = combined_params["beta"]
            self.td_epsilon = combined_params["td_epsilon"]

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> bool | None:

        # accumulate reward for n-step buffer
        if (self.buffer_config["useNstep"]):
            buf_entry = (state, action, reward, next_state, done)
            self.n_step_buf.append(buf_entry)
            if len(self.n_step_buf) < self.n_step:
                # not enough entries yet to calculate reward for the oldest element in the queue
                return False
            # calculate reward for oldest element in queue
            reward = 0.0
            discount = 1
            # setup default final state, done (i.e. of the state of the element just added)
            next_state = self.n_step_buf[-1][3]
            done = self.n_step_buf[-1][4]
            # calculate reward for oldest element in queue by accumulating rewards and gamma forward
            for i in range(self.n_step):
                s, a, r, ns, d = self.n_step_buf[i]
                reward += discount * r
                discount *= self.gamma
                if d:
                    # to avoid training on next episode, need to check if episode of the oldest element in the queue is over
                    # in case it is, need to break
                    next_state = ns
                    done = True
                    break
            state, action, _, _, _ = (
                self.n_step_buf[0]
            )

        # adding stuff to the replay buffer
        self.state_buf[self.curr_ind] = state
        self.next_state_buf[self.curr_ind] = next_state
        self.acts_buf[self.curr_ind] = action
        self.rewards_buf[self.curr_ind] = reward
        self.done_buf[self.curr_ind] = done

        # storing index for prioritized replay buffer
        idx = self.curr_ind

        self.curr_ind = (
            (self.curr_ind + 1) % self.max_size
        )  # if at end, go back to start and replace the oldest experiences
        self.size = min(
            self.size + 1, self.max_size
            # buffer size increase (capped at max_size) -> replace oldest back in start
        )

        # adding priority to the sum and min trees
        if self.buffer_config["usePrioritized"]:
            # epsilon clears edge case: if priority -> 0
            new_priority = (self.max_priority + self.td_epsilon) ** self.omega
            self.sum_priority_tree[idx] = new_priority
            self.min_priority_tree[idx] = new_priority
            # if at end, go back to start (like curr_ind with buffer)
            self.tree_pointer = (self.tree_pointer + 1) % self.max_size

        return True

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """
        returns:
            dict with keys: (obs, next_obs, acts, rews, done)
        """
        assert len(
            # if not, then were not even ready to sample, what are you doing here?
            self) >= self.batch_size

        if self.buffer_config["usePrioritized"]:
            idxs = []
            p_total_mass = self.sum_priority_tree.apply_sum()
            segment_mass = p_total_mass / self.batch_size
            # for each segment, sample idx and append in []
            for i in range(self.batch_size):
                # ex. interval goes from [0,1] -> [0.0, 0.25]
                a = segment_mass * i
                b = segment_mass * (i+1)
                upperbound = np.random.uniform(a, b)
                idx = self.sum_priority_tree.retrieve(upperbound)
                idx = idx % self.max_size
                idxs.append(idx)

            weights = []
            for i in idxs:
                weight = self.calculate_IS_weights(i, self.beta)
                weights.append(weight)
            weights = np.array(weights)

            return {
                "obs": self.state_buf[idxs],
                "next_obs": self.next_state_buf[idxs],
                "acts": self.acts_buf[idxs],
                "rews": self.rewards_buf[idxs],
                "done": self.done_buf[idxs],

                # addtional params
                "weights": np.array(weights),
                "idxs": np.array(idxs),
            }
        if (
            self.size < self.batch_size
        ):  # buffer is not yet filled, sample with replacement
            idxs = np.random.choice(
                self.size, size=self.batch_size, replace=True)
        else:
            idxs = np.random.choice(
                self.size, size=self.batch_size, replace=False)

        return {
            "obs": self.state_buf[idxs],
            "next_obs": self.next_state_buf[idxs],
            "acts": self.acts_buf[idxs],
            "rews": self.rewards_buf[idxs],
            "done": self.done_buf[idxs],
        }

    def __len__(self) -> int:
        return min(self.size, self.max_size)

    def calculate_IS_weights(self, idx, beta):
        '''
        Calculate the Importance-Sampling (IS) weight of each sample.i.e Each sample might have a differnet importance.
        Parameters:
            idx = idx of sampled priority
            Beta = correction factor (0 means no correction, 1 means full corection)
                   beta will start near 0 to 1 as more samples happen (training)
        '''
        assert beta > 0

        priority_i = self.sum_priority_tree[idx]
        # ensures no div by 0
        probability_i = max(
            priority_i / self.sum_priority_tree.apply_sum(), 1e-10)
        w_i = (1/len(self) * 1/probability_i) ** beta

        min_priority = self.min_priority_tree.apply_min()
        # ensures not div by 0
        min_probability = max(
            min_priority / self.sum_priority_tree.apply_sum(), 1e-10)
        max_weight = (1/len(self) * 1/min_probability) ** beta

        return w_i / max_weight  # return normalized weight

    def update_priorities(self, idxs, new_priorities: np.ndarray):
        '''Given a list of idx with its associated properties, update buffer.
            Note: MUST do abs(priority) + epsilon before calling this function.
        '''
        assert len(
            idxs) == new_priorities.size  # ensures we have a valid 1 to 1 matching
        for i, p in zip(idxs, new_priorities):
            self.max_priority = max(p, self.max_priority)  # update max
            self.sum_priority_tree[i] = (p) ** self.omega
            self.min_priority_tree[i] = (p) ** self.omega


class PrioritizedReplayBufferReturn(TypedDict):
    obs: np.ndarray
    next_obs: np.ndarray
    acts: np.ndarray
    rews: np.ndarray
    done: np.ndarray
    weights: np.ndarray  # Important to use for loss:= weight * TD
    idxs: np.ndarray  # Important to use for update_priorities
