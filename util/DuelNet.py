from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class DuelNet(nn.Module):
    '''
    Instead of having one fc (fully-connected) stream : input -> (hidden layers) -> output -> Q(s,a),
    DuelNet seperates the full stream into two parts: a shared feature layer, and a branch of 2 layers.

                                 ->  value_layer = V(s)
    i.e. input -> (hidden layer)                          -> output -> Q(s,a) := V(s) + [A(s,a) - mean A(s,*)]
                                 ->  advantage = A(s,a)

    this is advantageous for environements where actions don't really have a correlatiion to value. they don't matter

    ex. In trading, what if our state is very flat, no need to buy nor sell. Well
    instead of spending time and stress deciding which of the "equally bad" actions to choose, we just isoalte V(s) and
    realize that none of the action values are special -> finally Q reflects well the "boring current state"
    '''

    def __init__(self, input_dim : Tuple[int,...], output_dim:int, hidden_dim=256):
        super().__init__()
        flat_input_dim = int(np.prod(input_dim))
        # shared layer
        self.fc1 = nn.Linear(flat_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Value Layer : V(s) -> scalar value (one)
        self.fc_value = nn.Linear(hidden_dim, 1)
        # Advantage Layer : A(s,a) = vector for each action (output_dim)
        self.fc_advantage = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        shared = F.relu(self.fc1(x))
        shared = F.relu(self.fc2(shared))
        # now split shared into two streams
        v = self.fc_value(shared)
        a = self.fc_advantage(shared)

        q = v + (a - a.mean(dim=-1, keepdim=True))
        return q
