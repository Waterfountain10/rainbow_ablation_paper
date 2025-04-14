from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class DuelNet(nn.Module):
    def __init__(self, input_dim : Tuple[int,...], output_dim:int, hidden_dim=256):
        '''
        Instead of having one fc (fully-connected) stream : input -> (hidden layers) -> output -> Q(s,a),
        DuelNet seperates the full stream into two parts: a shared feature layer, and a branch of 2 layers.

                                     ->  value_layer = V(s)
        i.e. input -> (hidden layer)                          -> output -> Q(s,a) := V(s) + [A(s,a) - meanA(s,*)]
                                     ->  advantage = A(s,a)

        this is advantageous for environements where actions don't really have a correlatiion to value. they don't matter

        ex. In trading, what if our state is very flat, no need to buy nor sell. Well
        instead of spending time and stress deciding which of the "equally bad" actions to choose, we just isoalte V(s) and
        realize that none of the action values are special -> finally Q reflects well the "boring current state"
        '''
