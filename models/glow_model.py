import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeLayer(nn.Module):
    def forward(self, x, reverse=False):
        B, C, H, W = x.shape
        if not reverse:
            # 正向：[B, 3, 64, 64] -> [B, 12, 32, 32]
            x = x.reshape(B, C, H // 2, 2, W // 2, 2)
            x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, H // 2, W // 2)
            return x
        else:
            # 反向：[B, 12, 32, 32] -> [B, 3, 64, 64]
            # 这是刚才漏掉的关键代码！
            x = x.reshape(B, C // 4, 2, 2, H, W)
            x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, C // 4, H * 2, W * 2)
            return x

class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w = torch.randn(num_channels, num_channels)
        q, _ = torch.linalg.qr(w)
        self.weight = nn.Parameter(q)

    def forward(self, x, reverse=False):
        if not reverse:
            _, logabsdet = torch.slogdet(self.weight.to("cpu"))
            logabsdet = logabsdet.to(x.device)
            return F.conv2d(x, self.weight[:, :, None, None]), logabsdet * x.size(2) * x.size(3)
        else:
            inv_weight = torch.linalg.inv(self.weight.to("cpu")).to(x.device)
            return F.conv2d(x, inv_weight[:, :, None, None])

class AffineCoupling(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(num_channels // 2, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, num_channels, 3, padding=1)
        )
        self.nn[-1].weight.data.zero_()

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        h = self.nn(x1)
        shift, scale = h.chunk(2, dim=1)
        scale = torch.sigmoid(scale + 2)
        if not reverse:
            y2 = x2 * scale + shift
            return torch.cat([x1, y2], dim=1), torch.sum(torch.log(scale), dim=[1, 2, 3])
        else:
            y2 = (x2 - shift) / scale
            return torch.cat([x1, y2], dim=1)

class SimplifiedGlow(nn.Module):
    def __init__(self, num_layers=8, in_channels=3):
        super().__init__()
        self.squeeze = SqueezeLayer()
        mid_channels = in_channels * 4
        self.layers = nn.ModuleList([
            nn.ModuleList([Invertible1x1Conv(mid_channels), AffineCoupling(mid_channels)])
            for _ in range(num_layers)
        ])

    def transform_to_noise(self, x):
        log_det = 0
        x = self.squeeze(x)
        for conv, coupling in self.layers:
            x, ld1 = conv(x)
            x, ld2 = coupling(x)
            log_det += (ld1 + ld2)
        
        z_s = x[:, :4, :, :]
        z_c = x[:, 4:, :, :]
        return z_s, z_c, log_det

    def reverse(self, z_s, z_c):
        z = torch.cat([z_s, z_c], dim=1)
        for conv, coupling in reversed(self.layers):
            z = coupling(z, reverse=True)
            z = conv(z, reverse=True)
        # 这里的 reverse=True 现在会正确地将 12 通道变回 3 通道
        return self.squeeze(z, reverse=True)