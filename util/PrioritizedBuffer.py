from ReplayBuffer import ReplayBuffer, ReplayBufferReturn
from typing import Dict, Tuple, TypedDict, cast, List
from SegmentTree import MinTree, SumTree
from dqn import DQN
import numpy as np
import gymnasium as gym
import gym_anytrading
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL


# the idea is that you sample transitiions based on their prioritized probability : P(i)
#
#          (priority_i ) ^ omega
# P(i) := ------------------------       where priority_i := |TD_i| + epsilon (keeping sampling where TD = 0)
#          sum_k(priority_k ^ omega)

# Also, we might notice that this will cause a overfitting problem:
# at first especially, there will be high TD, so system will overfit the same initial transitions
# To combat this, it uses stochastic sampling (a mix of greedy prioritization AND uniform)
#
# however now, when it switches to uniform (its not really uniform, since past samples have been corrupted with greedy prioritization)
# this makes uniform (not a pure uniform but a rather biased unifrom sampling)
#
# and to fix this:  Importance-Sampling (IS) weights:
#
#   w_i = (1/N * 1/P(i))^Beta           where Beta := correction parameter (starts small -> 1 at end)
#
# that ensures that transitions that are being used a LOT, bigger P(i) for greedy prioritiation,
# gets dampened or has smaller weights
#
# eventually we use it for the loss := w_i * TD_i

# We will use segment trees since they have O(logn) lookups AND updates... and heaps only focuses on min or max not i
# - one sum_tree (queries sum for any range quickly)
# - one min_tree (helps track the minimum priority in O(logn)

# in short:
# sum_tree -> P(i) and
# min_tree -> min_priority -> max_w -> normalized_w_i
class PrioritizedReplayBufferReturn(TypedDict):
    obs: np.ndarray
    next_obs: np.ndarray
    acts: np.ndarray
    rews: np.ndarray
    done: np.ndarray
    weights: np.ndarray # Important to use for loss:= weight * TD
    idxs: np.ndarray # Important to use for update_priorities

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, obs_shape: Tuple[int, ...], size: int, batch_size: int = 32, omega = 0.6, beta = 0.6, td_epsilon = 1e-6):
        super().__init__(obs_shape, size, batch_size)
        # Create 2 trees to track sum_priorities, and minimum_priority:
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2
        self.sum_priority_tree = SumTree(tree_capacity)
        self.min_priority_tree = MinTree(tree_capacity)
        self.max_priority = 1.0 # initial max must be 1, then itll shift to 0.7 or wtv
        self.tree_pointer = 0

        self.omega = omega
        self.beta = beta
        self.td_epsilon = 1e-6 # same one in original PER paper (Schaul et al. 2015)

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        '''
        Same as regular replay buffer, but since we're adding a NEW transition, we add
        priority_t = MAX_PRIORITY, so that, later when sampling, that probability_t will have the HIGHEST prob"

        Return: void, but...
            - stores new priority in both trees AND shifts tree_pointer
            - updates buffer like regular ReplayBuffer
        '''
        super().store(state, action, reward, next_state, done) # store exp. + move buffer_idx
        new_priority = (self.max_priority + self.td_epsilon) ** self.omega # epsilon clears edge case: if priority -> 0
        self.sum_priority_tree[self.tree_pointer] = new_priority
        self.min_priority_tree[self.tree_pointer] = new_priority
        self.tree_pointer = (self.tree_pointer + 1) % self.max_size # if at end, go back to start (like curr_ind with buffer)

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
        probability_i = priority_i / self.sum_priority_tree.apply_sum()
        w_i = (1/len(self) * 1/probability_i) ** beta

        min_priority = self.min_priority_tree.apply_min()
        min_probability = min_priority / self.sum_priority_tree.apply_sum()
        max_weight = (1/len(self) * 1/min_probability) ** beta

        return w_i / max_weight # return normalized weight


    # overrides regular uniform sample
    def sample_batch(self) -> PrioritizedReplayBufferReturn:
        '''
        Same thing as before but with two additional output params.
        Also, rather than choosing transitions with uniform Prob, we retrieve with prioritized Prob.

        Returns: dict with (keys obs, next_obs, acts, rews, done, weights, indices)

        Ex: imagine 4 transitions stored. Regular ReplayBuffer -> each i=0,...i=3 has 25% chance
            but with PER and priorities = [0.1,0.3,0.2,0.4] -> P(0)=10%, P(1)=30%, ... P(3)=40%
        '''
        assert len(self) >= self.batch_size # if not, then were not even ready to sample, what are you doing here?

        # Idea: Stratified sampling https://en.wikipedia.org/wiki/Stratified_sampling
            # Calculate the total priority mass (sum of all priorities)
            # Then, we segment partition that mass into x amount of segments (x = batch_size here)
            # for each segment [a, b), we sample a random point with ~Unif(a,b), retrieve the idx from sum_tree
            # this method ensures proportional prioritization is preserved (higher priorit = larger interval (a,b))

        idxs = []
        p_total_mass = self.sum_priority_tree.apply_sum()
        segment_mass = p_total_mass / self.batch_size
        for i in range(self.batch_size): # for each segment, sample idx and append in []
            a = segment_mass * i     # ex. interval goes from [0,1] -> [0.0, 0.25]
            b = segment_mass * (i+1)
            upperbound = np.random.uniform(a,b)
            idx = self.sum_priority_tree.retrieve(upperbound)
            idxs.append(idx)

        weights = []
        for i in idxs:
            weight = self.calculate_IS_weights(i, self.beta)
            weights.append(weight)
        weights = np.array(weights)

        return cast(PrioritizedReplayBufferReturn, {
            "obs": self.state_buf[idxs],
            "next_obs": self.next_state_buf[idxs],
            "acts" :self.acts_buf[idxs],
            "rews" : self.rewards_buf[idxs],
            "done" : self.done_buf[idxs],
            "weights" : np.array(weights),
            "idxs" : idxs,
        })

    def __len__(self) -> int:
        return self.size

    def update_priorities(self, idx, priorities: np.ndarray):
        '''Given a list of idx with its associated properties, update buffer.
            Note: MUST do abs(priority) before calling this function.
                  No need for epsilon pre-call (taken care here)
        '''
        assert len(idx) == len(priorities) # ensures we have a valid 1 to 1 matching
        for i, p in zip(idx, priorities):
            self.max_priority = max(p, self.max_priority) # update max
            self.sum_priority_tree[i] = (p + self.td_epsilon) ** self.omega
            self.min_priority_tree[i] = (p + self.td_epsilon) ** self.omega

if __name__ == "__main__":
    # EXAMPLE Test run with Prioritized Buffer and DQN

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
    print(np.mean(rewards))
