from unittest.mock import right
from xml.dom import IndexSizeErr
import operator as op

class SegmentTree:
    def __init__(self, capacity:int, function):
        self.capacity = capacity
        self.tree = []
        self.function = function
        '''
        ex. cap = 4,  tree[cap*2] = tree[8]

                            [0 to 3] i=1
                            /             \
                [0 to 1] i=2                [2 to 3] i=3
                    /       \               /           \
        [0 to 0] i=4    [1 to 1] i=5   [2 to 2] i=6    [3 to 3] i=7
        '''

    def __setitem__(self, idx, val):
        '''Sets item in tree with index. ex: tree[idx] = value'''
        idx += self.capacity # array gets bigger which also changes tree
        self.tree[idx] = val
        # reformat the nodes above (update sum or update minimum value now)
        idx = idx // 2
        while idx > 0:
            self.tree[idx] = self.function( # recompute the value of parent with both kids
                self.tree[idx*2],
                self.tree[idx*2+1]
            )
            idx = idx // 2 # do to parent

    def __getitem__(self, idx):
        '''Get value of tree[idx] = value at its corresponding leaf.
        ex: value = tree[6], we need to find the leaf node with sum/min of itself
        '''
        if idx < 0:
            raise IndexSizeErr("Sorry, tree index is out of bound (negative).")
        elif idx >= self.capacity:
            raise IndexSizeErr("Sorry, tree index is out of bound (over capacity).")
        else:
            return self.tree[self.capacity + idx] # offset to get the leaf's value


    def apply_function(self, range_start = 0, range_end = 0):
        '''Return the result of applying function (either min or sum) to a subsequence of array (self.tree).'''
        if range_end <= 0:
            range_end += self.capacity
        range_end -= 1
        return self.rec_apply_function(range_start, range_end, 1, 0, self.capacity - 1)

    def rec_apply_function(self, range_start, range_end, i, left_ptr, right_ptr):
        '''Helper for apply_function'''
        # base case : if query has perfect match, return node_index's value
        if range_start == left_ptr and range_end == right_ptr:
            return self.tree[i]
        mid = (left_ptr + right_ptr) // 2 # ex. 2 + 5 // 2 = 3

        # 1) Query range is all in left side (mid included)
        if range_end <= mid:
            return self.rec_apply_function(range_start, range_end, 2*i, left_ptr, mid)

        # 2) Query range is all in right side (mid exlcuded)
        elif mid + 1 <= range_start:
            return self.rec_apply_function(range_start, range_end, 2*i+1, mid + 1, right_ptr )

        # 3) Query range is in both sides (ex. mid = 5, range is 4 -> 7)
        else:
            return self.function(
                self.rec_apply_function(range_start, mid, 2*i, left_ptr,mid),
                self.rec_apply_function(mid + 1, range_end, 2*i+1, mid + 1, right_ptr)
            )


class SumTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity = capacity, function = op.add)
        for _ in range(2 * self.capacity): # initialize tree with 0-s
            self.tree.append(0.0)

    def apply_sum(self, range_start = 0, range_end = 0):
        return super().apply_function(range_start, range_end)

    def retrieve(self, upperbound):
        '''
        Finds the max idx (right below upperbound) in tree.
        Upperbound is a float from 0 to 1.0
        '''
        idx = 1 # start at root
        while idx < self.capacity:
            left_child = 2 * idx
            right_child = left_child + 1
            if self.tree[left_child] > upperbound: # if left child is already higher than upperbound, then right CANT be better
                idx = 2 * idx
            else: # go right
                upperbound -= self.tree[left_child] # take out sum of left
                idx = right_child
        return idx - self.capacity # to get the "simple index" and not the actual index of leafs


class MinTree(SegmentTree):
    '''This tree is ONLY used for calculate_IS_weights because it helps find max_weight through min_priority.
        When trying to find priority_tree[idx], ONLY refer to SUM_TREE instead, not this one!
    '''

    def __init__(self, capacity):
        super().__init__(capacity = capacity, function = min)
        for _ in range(2 * self.capacity): # initialize tree with float(inf)
            self.tree.append(float("inf"))

    def apply_min(self, range_start = 0, range_end = 0):
        return super().apply_function(range_start, range_end)


if __name__ == "__main__":
    CAPACITY = 4
    OMEGA = 0.6
    BETA = 0.4
    priorities = [0.1, 0.3, 0.2, 0.4]

    '''
    ex. cap = 4,  tree[cap*2] = tree[8]

                        [0 to 3] i=1
                        /             \
            [0 to 1] i=2                [2 to 3] i=3
                /       \               /           \
    [0 to 0] i=4    [1 to 1] i=5   [2 to 2] i=6    [3 to 3] i=7
        0.1             0.3             0.2             0.4
    '''

    sum_tree = SumTree(CAPACITY)
    min_tree = MinTree(CAPACITY)

    # fill in the trees
    for i, p in enumerate(priorities):
        sum_tree[i] = p
        min_tree[i] = p

    # debug and print tree contents
    print("sum_tree:", sum_tree.tree) # sum_tree: [0.0, 1.0, 0.4, 0.6000000000000001, 0.1, 0.3, 0.2, 0.4]
    print("min_tree:", min_tree.tree) #  min_tree: [inf, 0.1, 0.1, 0.2, 0.1, 0.3, 0.2, 0.4]
    # yes, i = 0 is supposed to be default value as root is i = 1

    print("total sum:", sum_tree.apply_sum()) # total sum: 1.0
    print("total min:", min_tree.apply_min()) # total min: 0.1
    print("left sum:", sum_tree.apply_sum(0,2)) # left sum: 0.4 (we dont include idx = 2)

    # update
    sum_tree[2] = 0.9 # update the 3rd leaf from the right
    print("updated sum tree:", sum_tree.tree)
    # updated sum tree: [0.0, 1.7000000000000002, 0.4, 1.3, 0.1, 0.3, 0.9, 0.4]
    #  we see i = 6 (updated leaf) : 0.2 -> 0.9, so added +0.7
    #         i = 3 (right parent), 0.6 + 0.7 = 1.3
    #         i = 1 (grandparent), 1.0 + 0.7 = 1.7
    print("updated new sum:", sum_tree.apply_sum()) # 1.7000000000000002
