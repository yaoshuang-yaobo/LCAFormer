import random

import torch
from torch import nn
from thop import profile
from model.blocks.LDConv import *
from torch.nn import ChannelShuffle


class LCAM(nn.Module):
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.dim = dim
        self.LDConv = LDConv(2, self.dim, num_param = kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        GAP = torch.mean(x, dim=1, keepdim=True)
        GMP = torch.max(x, dim=1, keepdim=True)[0]
        out  = torch.cat([GAP, GMP], dim=1)
        out = self.sigmoid(self.LDConv(out)) * x + x

        return out



