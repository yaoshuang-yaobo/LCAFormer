import random

import torch
from torch import nn
from thop import profile
from torch.nn import ChannelShuffle

from model.blocks.BAM import *

def channel_shuffle(x, groups):

    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # num_channels = groups * channels_per_group

    # grouping, 通道分组
    # b, num_channels, h, w =======>  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()
    # x.shape=(batchsize, channels_per_group, groups, height, width)
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class SSIU(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.dim_ = dim
        self.dim = dim//3
        self.dilation = dilation
        self.pad = int(self.dilation - 1)

        self.dw = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(self.dim, self.dim, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.ddw = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=(3, 1), padding=(self.pad, 1), dilation=self.dilation),
            nn.Conv2d(self.dim, self.dim, kernel_size=(1, 3), padding=(1, self.pad), dilation=self.dilation),
        )
        self.BAM = BAM(self.dim)
        self.bam = BAM(self.dim_)

    def forward(self, x):
        x_ = x
        x1, x2, x3 = x.split(self.dim_//3, dim=1)

        x1 = self.BAM(self.dw(x1))
        x2 = self.BAM(x2)
        x3 = self.BAM(self.ddw(x3))

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.bam(x)
        x = x * x_ + x_

        x = channel_shuffle(x, 3)

        return x


