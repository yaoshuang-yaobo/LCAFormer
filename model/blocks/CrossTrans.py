import random

import torch
from torch import nn
from thop import profile
from torch.nn import ChannelShuffle


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(a, b , ks, stride, pad, dilation, groups, bias=False),
            nn.BatchNorm2d(b),
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class Attention(torch.nn.Module):
    def __init__(self, dim, dim_s, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None, ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim_s, nh_kd, 1)
        self.to_k = Conv2d_BN(dim, nh_kd, 1)
        self.to_v = Conv2d_BN(dim, self.dh, 1)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim_s, bn_weight_init=0))

    def forward(self, x, singlex):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        _, C_s, H_s, W_s = get_shape(singlex)

        qq1 = self.to_q(singlex)
        qq2 = qq1.reshape(B, self.num_heads, self.key_dim, H_s * W_s)
        qq = qq2.permute(0, 1, 3, 2)
        kk1 = self.to_k(x)
        kk = kk1.reshape(B, self.num_heads, self.key_dim, H * W)
        vv1 = self.to_v(x)
        vv2 = vv1.reshape(B, self.num_heads, self.d, H_s * W_s)
        vv = vv2.permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k

        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx

class CrossTrans(nn.Module):

    def __init__(self, dim, key_dim=16, num_heads=6, mlp_ratio=4., attn_ratio=2.,
                 drop_path=0., act_layer=nn.ReLU):
        super().__init__()
        self.dim = dim
        self.dim_s = self.dim // 4
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Attention(dim, dim_s=self.dim_s, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio,
                              activation=act_layer)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.proj = torch.nn.Sequential(Conv2d_BN(
            self.dim_s, dim, bn_weight_init=0),
           nn.ReLU(inplace=True), )

    def forward(self, xx):
        x1, x2, x3, x = xx.split(self.dim//4, dim=1)
        x = x + self.drop_path(self.attn(xx, x))
        x = self.proj(x)
        return x




