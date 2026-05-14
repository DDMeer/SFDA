import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeLayer(nn.Module):
    def forward(self, x, reverse=False):
        B, C, H, W = x.shape
        if not reverse:
            # [B, C, H, W] -> [B, C*4, H//2, W//2]
            x = x.reshape(B, C, H // 2, 2, W // 2, 2)
            x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, H // 2, W // 2)
        else:
            # 逆向还原 [B, C, H, W] -> [B, C/4, H*2, W*2]
            x = x.reshape(B, C // 4, 2, 2, H, W)
            x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, C // 4, H * 2, W * 2)
        return x

class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # 使用最新的 linalg.qr 替换已过时的 torch.qr
        w = torch.randn(num_channels, num_channels)
        q, _ = torch.linalg.qr(w)
        self.weight = nn.Parameter(q)

    def forward(self, x, reverse=False):
        # x shape: [B, C, H, W]
        if not reverse:
            # 方案 A 核心修改：将权重移至 CPU 计算 slogdet，避开 MPS 算子缺失问题
            # slogdet 返回 (sign, logabsdet)，我们只需要 logabsdet
            _, logabsdet = torch.slogdet(self.weight.to("cpu"))
            logabsdet = logabsdet.to(x.device)
            
            return F.conv2d(x, self.weight[:, :, None, None]), logabsdet * x.size(2) * x.size(3)
        else:
            # 逆向转换：求逆运算也移至 CPU 确保精度和兼容性
            inv_weight = torch.linalg.inv(self.weight.to("cpu")).to(x.device)
            return F.conv2d(x, inv_weight[:, :, None, None])

class AffineCoupling(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # 内部网络无需可逆，使用常规 CNN
        self.nn = nn.Sequential(
            nn.Conv2d(num_channels // 2, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, num_channels, 3, padding=1)
        )
        # 最后一层初始化为 0，确保模型刚开始训练时是恒等变换，增加稳定性
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
        # 经过 Squeeze 后空间减半，通道 *4 (3 -> 12)
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
        return x, log_det

    def reverse(self, z):
        for conv, coupling in reversed(self.layers):
            z = coupling(z, reverse=True)
            z = conv(z, reverse=True)
        return self.squeeze(z, reverse=True)