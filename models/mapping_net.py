import torch
import torch.nn as nn

class StyleMapper(nn.Module):
    def __init__(self, zs_channels, img_size=64, sd_dim=768):
        super().__init__()
        # zs_dim = 像素级维度，通常很大
        input_dim = zs_channels * img_size * img_size
        
        self.mapper = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, sd_dim),
            nn.LayerNorm(sd_dim) # 对齐 CLIP 的 Norm 风格
        )

    def forward(self, zs):
        # zs: [B, C_s, H, W]
        x = zs.view(zs.size(0), -1)
        return self.mapper(x)