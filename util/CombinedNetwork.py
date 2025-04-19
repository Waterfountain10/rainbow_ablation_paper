from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class CombinedNeuralNet(nn.Module):
    def __init__(self, input_dim: Tuple[int, ...], output_dim: int, hidden_dim=512, network_config={
        "useDuel": False,
        "useNoisy": False,
        "useDistributive": False,
    },
        combined_params={
        # NoisyNet params
        "sigma_init": 0.9,

        # DistributiveNet params
        "atom_size": 51,
        "support": None,
    },
    ):

        super().__init__()

        self.network_config = network_config

        if self.network_config["useDistributive"]:
            self.support = combined_params["support"]
            self.atom_size = combined_params["atom_size"]

        # setting output dim to account for atom size if distributive net is used
        self.output_dim = output_dim * \
            self.atom_size if self.network_config["useDistributive"] else output_dim
        flat_input_dim = int(np.prod(input_dim))
        self.fc1 = nn.Linear(flat_input_dim, hidden_dim)

        # use noisy layers if specified otherwise use regular linear layers
        if self.network_config["useNoisy"]:
            self.sigma_init = combined_params["sigma_init"]
            self.fc2 = NoisyLayer(hidden_dim, hidden_dim, self.sigma_init)
            self.fc3 = NoisyLayer(hidden_dim, self.output_dim, self.sigma_init)
        else:
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, self.output_dim)

        if self.network_config["useDuel"]:
            self.fc_value = nn.Linear(hidden_dim, 1)


    def forward(self, x, return_prob=False):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)

        # DuelNet, q is the advantage
        if self.network_config["useDuel"]:
            v = self.fc_value(x)
            q = v + (q - q.mean(dim=-1, keepdim=True))

        if self.network_config["useDistributive"]:
            # Reshape to (batch_size, output_dim, atom_size)
            q = q.view(q.size(0), -1, self.atom_size)
            # Get probability distribution and clamp for numerical stability
            prob = F.softmax(q, dim=-1)
            prob = prob.clamp(min=1e-3)
            # get weighted sum by summing over the atoms
            if (isinstance(self.support, torch.Tensor)):
                q = torch.sum(prob * self.support, dim=-1)
            return (prob, q) if return_prob else q

        return q

    def reset_noise(self):
        ''' Updates the epsilon after each step. Called inside agent's update_model() (usually right after optimizer.step()).'''
        if (self.network_config["useNoisy"]):
            if isinstance(self.fc2, NoisyLayer):
                self.fc2.create_new_epsilon()
            if isinstance(self.fc3, NoisyLayer):
                self.fc3.create_new_epsilon()


class NoisyLayer(nn.Module):
    '''
    Instead of regular Linear Layer : y = Wx + b, where Weight W and bias b are both fixed.
    We add some randomness inside the Network (i.e the randomness is smarter, and knows when to be deterministic)

    -> Noisy Layer : y = base_y + trainable_y
                       = [W_base * x + b_base] + [ (W_trainable * eps_w) * x + (b_trainable * eps_b) ]
                       = (W_base + (W_trainable * epsi_w)) * x + (b_base + b_trainable * epsi_b)
                       = W'x + b' <-- eventually put a linear() on this

    Parameters (mu and sigma): all tensors with gradients, must allow backprop
        - mu or mean = base
        - sigma or std var = learnable noise Scale
            * these get learned

    Noise: (random sampled variable)
        - epsilon = random noise drawn from N(0,1)
            * these isnt learned
    '''

    def __init__(self, feature_input_size, feature_output_size, sigma_init):
        super().__init__()
        self.feature_input_size = feature_input_size
        self.feature_output_size = feature_output_size

        # initialize mu matrices in form : [output, input] for w, [output] for b
        self.w_mu = nn.Parameter(torch.Tensor(
            feature_output_size, feature_input_size))
        self.b_mu = nn.Parameter(torch.Tensor(feature_output_size))

        # initialize sigma matrices in form : [output, input] for w, [output] for b
        self.w_sigma = nn.Parameter(torch.Tensor(
            feature_output_size, feature_input_size))
        self.b_sigma = nn.Parameter(torch.Tensor(feature_output_size))
        # this will be useful for filling the above sigma matrices
        self.sigma_init = sigma_init

        # initialize epsilons as BUFFERS and NOT PARAMETERS since are not trying to learn from epsilons
        self.register_buffer("w_epsilon", torch.Tensor(
            feature_output_size, feature_input_size))
        self.register_buffer("b_epsilon", torch.Tensor(feature_output_size))

        # now that tensors are initialized, fill with respecitve values
        self.create_new_mu_and_sigma()
        self.create_new_epsilon()

    def create_new_mu_and_sigma(self):
        '''Pick new mean (mu) and std_var (sigma).
            Use Gaussian Transformation (see below more detail)

            According to NoisyNet authors, each element mu is initialized by sampling from Unif[-1/sqrt(input), 1/sqrt(input)]
            and for sigma, each is initialized const := sigma_init/sqrt(input)
            So, we will do that too
        '''
        # fill mu with uniform
        mu_unif_range = 1 / np.sqrt(self.feature_input_size)
        self.w_mu.data.uniform_(-1 * mu_unif_range, mu_unif_range)
        self.b_mu.data.uniform_(-1 * mu_unif_range, mu_unif_range)

        # fill sigmas with constant := sigma_init/sqrt(input)
        constant = self.sigma_init / np.sqrt(self.feature_input_size)
        self.w_sigma.data.fill_(constant)
        self.b_sigma.data.fill_(constant)

    def create_new_epsilon(self):
        '''
        Pick new noise (epsilon).
        Instead of using traditional sampling with N(0,1) which would take (input_size * output_size) amount of samples,
        the authors recommend factorized gaussian : creating a noise sample matrix by combining ONLY 2 1D vectors : epsilon_in and epsilon_out
        the latter is more cost-efficient since we are just using (input_size + output_size) amount of samples!

        Nonlinear Gaussian transformation; f(x) = sign(x) * sqrt(abs(x)) for random epsilon of size x
            -> epsilon_weight = matrix made by out product of f(input) * f(output)
            -> epsilon_bias = 1D matrix by f(output)
        '''
        epsilon_in = self._f(self.feature_input_size)
        epsilon_out = self._f(self.feature_output_size)

        # update weight epsilon field
        # outprduct to get W_matrix = [e_out, e_in] so input is row
        w_matrix = torch.outer(epsilon_out, epsilon_in)
        self.w_epsilon: torch.Tensor  # purely just to take out weird static type highlighting
        self.w_epsilon.copy_(w_matrix)

        # update bias epislon field
        self.b_epsilon: torch.Tensor  # purely just to take out weird static type highlighting
        self.b_epsilon.copy_(epsilon_out)

    def forward(self, x):
        '''
        Forward computation for NoisyLayer() is y':= W'x + b'
                =  (W_base + (W_trainable * epsi_w)) * x + (b_base + b_trainable * epsi_b)
        '''
        new_w = self.w_mu + (self.w_sigma * self.w_epsilon)
        new_b = self.b_mu + (self.b_sigma * self.b_epsilon)

        # linear = w'* x + b'
        return F.linear(x, new_w, new_b)

    # helper for create_new_epsilon

    def _f(self, size):
        '''Returns a torch tensor for f(x) = sign(x) * sqrt(|x|)'''
        x = torch.randn(size)  # get x = sized sample from N(0,1)
        sign_x = x.sign()
        sqrt_abs_x = x.abs().sqrt()
        return sign_x.mul(sqrt_abs_x)
