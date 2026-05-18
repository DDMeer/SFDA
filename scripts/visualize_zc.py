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
        # 提取底座 z_c
        outputs = glow.transform_to_noise(x)
        # 根据之前的报错，z_c 是 outputs 中的第一个 4 维张量
        z_tensors = [t for t in outputs if isinstance(t, torch.Tensor) and t.dim() == 4]
        z_c = z_tensors[0] # [1, 4, 32, 32]

    # 3. 绘制 4 个通道的热力图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Visualizing $z_c$ (The Content Base) Channels', fontsize=16)

    for i in range(4):
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