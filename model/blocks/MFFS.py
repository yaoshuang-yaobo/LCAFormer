import random
import torch.nn.functional as F
import torch
from torch import nn
from thop import profile

class MFFS(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

        self.q = nn.Sequential(
            nn.Conv2d(self.dim_1, self.dim_1, 1),
            nn.BatchNorm2d(self.dim_1),
            nn.ReLU(inplace=True),
        )
        self.k = nn.Sequential(
            nn.Conv2d(self.dim_1, self.dim_1, 1),
            nn.BatchNorm2d(self.dim_1),
            nn.ReLU(inplace=True),
        )
        self.v = nn.Sequential(
            nn.Conv2d(self.dim_1, self.dim_1, 1),
            nn.BatchNorm2d(self.dim_1),
            nn.ReLU(inplace=True),
        )

        self.sigmoid = nn.Sigmoid()

        self.change = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.dim_2, self.dim_1, 1),
            nn.BatchNorm2d(self.dim_1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        B, C, H, W = x1.size()
        x2_ = self.change(x2)

        x1_q = self.q(x1)
        x1_q = x1_q.reshape(B, -1, H*W).permute(0, 2, 1).contiguous()
        x1_k = self.k(x1)
        x1_k = x1_k.reshape(B, -1, H*W)
        x1_v = self.v(x1)
        x1_v = x1_v.reshape(B, -1, H*W).permute(0, 2, 1).contiguous()

        x2_q = self.q(x2_)
        x2_q = x2_q.reshape(B, -1, H*W).permute(0, 2, 1).contiguous()
        x2_k = self.k(x2_)
        x2_k = x2_k.reshape(B, -1, H*W)
        x2_v = self.v(x2_)
        x2_v = x2_v.reshape(B, -1, H*W).permute(0, 2, 1).contiguous()

        A1 = torch.matmul(x1_q, x2_k)
        A1 = self.sigmoid(A1)
        A2 = torch.matmul(x2_q, x1_k)
        A2 = self.sigmoid(A2)

        a1 = torch.matmul(A1, x1_v)
        a1 = a1.permute(0, 2, 1).contiguous().reshape(B, -1, H, W)
        a2 = torch.matmul(A2, x2_v)
        a2 = a2.permute(0, 2, 1).contiguous().reshape(B, -1, H, W)
        a = self.sigmoid(a1 + a2)

        X1 = a * x1
        X2 = a * x2_

        out = X1 + X2

        return out




