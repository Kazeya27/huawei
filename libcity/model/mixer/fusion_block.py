from pdb import set_trace

import torch
import math
from torch import nn


class FusionBlock(nn.Module):
    def __init__(self, input_window, t_patch_size, n_tlayer, act=nn.ReLU):
        super(FusionBlock, self).__init__()
        self.n_tlayer = n_tlayer
        self.act = act
        self.t_patch_size = t_patch_size
        window = input_window
        self.fusion_layers = nn.ModuleList()
        for i in range(n_tlayer):
            self.fusion_layers.append(nn.Sequential(
                nn.Linear(window, window),
                self.act,
                nn.Linear(window, window),
            ))


    def forward(self, x_list):
        for i in range(self.n_tlayer, 0, -1):
            x = x_list[i]
            x = self.fusion_layers[i - 1](x)
            x_list[i - 1] = x + x_list[i - 1]
        return x_list[0]

