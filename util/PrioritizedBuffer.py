from ReplayBuffer import ReplayBuffer, ReplayBufferReturn
from typing import Dict, Tuple, TypedDict
from SegmentTree import MinTree, SumTree
import numpy as np

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


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, obs_shape: Tuple[int, ...], size: int, batch_size: int = 32, omega = 0.6):
        super().__init__(obs_shape, size, batch_size)
        # PER specfici parameters:
        self.omega = omega
        self.max_priority = 1.0 # initial max priority score

        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2
        self.sum_priority_tree = SumTree(tree_capacity)
        self.min_prioirty_tree = MinTree(tree_capacity)
        return

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        '''Store() is same as regular replay buffer, but we're also adding priority'''
        super().store(state, action, reward, next_state, done)

    # overrides regular uniform sample
    def sample_batch(self) -> ReplayBufferReturn:



        return batch
