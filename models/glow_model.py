import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 基础可逆 1x1 卷积
class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # 初始化为一个随机旋转矩阵
        w = torch.randn(num_channels, num_channels)
        q, _ = torch.qr(w) 
        self.weight = nn.Parameter(q)

    def forward(self, x, reverse=False):
        # x shape: [B, C, H, W]
        if not reverse:
            return F.conv2d(x, self.weight[:, :, None, None]), torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)
        else:
            return F.conv2d(x, self.weight.inverse()[:, :, None, None])

# 仿射耦合层 (核心解耦模块)
class AffineCoupling(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # 这里的网络可以是任何复杂的 CNN，它不需要是可逆的
        self.nn = nn.Sequential(
            nn.Conv2d(num_channels // 2, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, num_channels, 3, padding=1)
        )
        self.nn[-1].weight.data.zero_() # 初始化为 0 保证训练初期是恒等变换

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1) # 拆分通道
        h = self.nn(x1)
        shift, scale = h.chunk(2, dim=1)
        scale = torch.sigmoid(scale + 2) # 缩放因子

        if not reverse:
            y2 = x2 * scale + shift
            return torch.cat([x1, y2], dim=1), torch.sum(torch.log(scale), dim=[1, 2, 3])
        else:
            y2 = (x2 - shift) / scale
            return torch.cat([x1, y2], dim=1)

# 完整的 Glow 简化版结构
class SimplifiedGlow(nn.Module):
    def __init__(self, num_layers=12, num_channels=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([Invertible1x1Conv(num_channels), AffineCoupling(num_channels)])
            for _ in range(num_layers)
        ])

    def transform_to_noise(self, x):
        log_det = 0
        for conv, coupling in self.layers:
            x, ld1 = conv(x)
            x, ld2 = coupling(x)
            log_det += (ld1 + ld2)
        return x, log_det

    def reverse(self, z):
        for conv, coupling in reversed(self.layers):
            z = coupling(z, reverse=True)
            z = conv(z, reverse=True)
        return z