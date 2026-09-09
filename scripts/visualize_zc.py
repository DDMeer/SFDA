import torch
import os
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import numpy as np

# 补丁：处理 Mac 环境
if not hasattr(torch, 'xpu'):
    class Mock:
        def __getattr__(self, name): return lambda *args, **kwargs: None
    torch.xpu = Mock()

sys.path.append(os.getcwd())
from models.glow_model import SimplifiedGlow

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def visualize_zc():
    # 1. 加载模型
    glow = SimplifiedGlow().to(device)
    glow.load_state_dict(torch.load('checkpoints/glow_stage1.pth', map_location=device))
    glow.eval()

    # 2. 读取测试图 63.jpg
    img_path = "data/art_painting/0/63.jpg"
    img = Image.open(img_path).convert('RGB')
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # 提取底座 z_c：显式解包 (z_s, z_c, log_det)
        z_s, z_c, _ = glow.transform_to_noise(x)
        assert z_s.shape[1] == 4, f"z_s 应为 4 通道 (style)，实得 {z_s.shape[1]}"
        assert z_c.shape[1] == 8, f"z_c 应为 8 通道 (content)，实得 {z_c.shape[1]}"
        # z_c: [1, 8, 32, 32]

    # 3. 绘制 z_c 全部通道的热力图
    num_ch = z_c.shape[1]
    fig, axes = plt.subplots(1, num_ch, figsize=(2.5 * num_ch, 5))
    fig.suptitle('Visualizing $z_c$ (The Content Base) Channels', fontsize=16)

    for i in range(num_ch):
        channel_data = z_c[0, i].cpu().numpy()
        # 归一化便于观察
        channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)
        
        im = axes[i].imshow(channel_data, cmap='viridis')
        axes[i].set_title(f'Channel {i}')
        axes[i].axis('off')

    plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    plt.savefig('zc_channels_visualization.png')
    print("✅ 可视化报告已生成: zc_channels_visualization.png")
    plt.show()

if __name__ == "__main__":
    visualize_zc()