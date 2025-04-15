import torch
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RewardsNeuralNet(nn.Module):
    def __init__(self, input_dim: Tuple[int, ...], output_dim: int, atom_size: int, support: torch.Tensor, hidden_dim=128):
        super(RewardsNeuralNet, self).__init__()
        
        flat_input_dim = int(np.prod(input_dim))
        self.fc1 = nn.Linear(flat_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim * atom_size)
        
        # Distributional DQN network parameters
        self.atom_size = atom_size
        self.support = support
        self.temp = 1.0

        # Initialize neural net
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)  # Better than uniform
            nn.init.constant_(layer.bias, 0.01)

    def forward(self, x, return_prob=False):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(x.size(0), -1, self.atom_size)  # reshape to (batch_size, output_dim, atom_size)
        prob = F.softmax(x/self.temp, dim=-1)  # get probability distribution
        prob = prob.clamp(min=1e-3) # clamp for numerical stability
        q = torch.sum(prob * self.support, dim=-1)  # get weighted sum by summing over the atoms
        return (prob, q) if return_prob else q







