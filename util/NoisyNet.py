import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

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
        self.w_mu = nn.Parameter(torch.Tensor(feature_output_size, feature_input_size))
        self.b_mu = nn.Parameter(torch.Tensor(feature_output_size))

        # initialize sigma matrices in form : [output, input] for w, [output] for b
        self.w_sigma = nn.Parameter(torch.Tensor(feature_output_size, feature_input_size))
        self.b_sigma = nn.Parameter(torch.Tensor(feature_output_size))
        self.sigma_init = sigma_init # this will be useful for filling the above sigma matrices

        # initialize epsilons as BUFFERS and NOT PARAMETERS since are not trying to learn from epsilons
        self.register_buffer("w_epsilon", torch.Tensor(feature_output_size, feature_input_size))
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
        w_matrix = torch.outer(epsilon_out, epsilon_in) # outprduct to get W_matrix = [e_out, e_in] so input is row
        self.w_epsilon : torch.Tensor # purely just to take out weird static type highlighting
        self.w_epsilon.copy_(w_matrix)

        # update bias epislon field
        self.b_epsilon : torch.Tensor # purely just to take out weird static type highlighting
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
        x = torch.randn(size) # get x = sized sample from N(0,1)
        sign_x = x.sign()
        sqrt_abs_x = x.abs().sqrt()
        return sign_x.mul(sqrt_abs_x)


class NoisyNet(nn.Module):
    '''Same thing as Regular Network, but we implement NoisyLayer() instead of regular Linear()
        This is essentially NeuralNet + NoisyLayers (i.e first is linear and 2 last layers are noisy)

            TO NOTE: rainbow authors also implemented DuelNet + NoisyLayers, which gave better results...
                     thus, I added an implementation of how it would look like with DuelNet, under the code
    '''
    def __init__(self, input_dim: Tuple[int, ...], ouput_dim: int, hidden_dim=256, sigma_init = 0.5):
        super().__init__()
        flat_input_dim = int(np.prod(input_dim))
        self.fc1 = nn.Linear(flat_input_dim, hidden_dim)
        self.fc2_noisy = NoisyLayer(hidden_dim, hidden_dim, sigma_init)
        self.fc3_noisy = NoisyLayer(hidden_dim, ouput_dim, sigma_init)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2_noisy(x))
        return self.fc3_noisy(x)

    def reset_noise(self):
        ''' Updates the epsilon after each step. Called inside agent's update_model() (usually right after optimizer.step()).'''
        self.fc2_noisy.create_new_epsilon()
        self.fc3_noisy.create_new_epsilon()
