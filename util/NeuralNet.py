import torch.nn as nn
import torch.nn.functional as F
import torch

class NeuralNet(nn.Module):
    def __init__(self, input_dim: int, ouput_dim: int, hidden_dim=256):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, ouput_dim)

        # TODO: how to init neural net?
        # for layer in [self.fc1, self.fc2, self.fc3]:
        #     layer.weight.data.uniform_(-0.001, 0.001)
        #     layer.bias.data.uniform_(-0.001, 0.001)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)