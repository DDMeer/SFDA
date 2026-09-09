import torch
import os
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image

# 补丁：处理 Mac 环境
if not hasattr(torch, 'xpu'):
    class Mock:
        def __getattr__(self, name): return lambda *args, **kwargs: None
    torch.xpu = Mock()

sys.path.append(os.getcwd())
from models.glow_model import SimplifiedGlow

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def verify():
    glow = SimplifiedGlow().to(device)
    glow.load_state_dict(torch.load('checkpoints/glow_stage1.pth', map_location=device))
    glow.eval()

    img_path = "data/art_painting/0/63.jpg"
    if not os.path.exists(img_path):
        print("❌ 找不到图片 63.jpg，请检查路径！")
        return

    img = Image.open(img_path).convert('RGB')
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # --- 显式解包：transform_to_noise 返回 (z_s, z_c, log_det) ---
        z_s, z_c, _ = glow.transform_to_noise(x)
        assert z_s.shape[1] == 4, f"z_s 应为 4 通道 (style)，实得 {z_s.shape[1]}"
        assert z_c.shape[1] == 8, f"z_c 应为 8 通道 (content)，实得 {z_c.shape[1]}"

        print(f"📊 提取成功: z_c 维度 {z_c.shape}, z_s 维度 {z_s.shape}")

        # --- 执行逆向重构 (zs 归零：抹掉风格纤维，保留内容底座) ---
        z_s_zero = torch.zeros_like(z_s)
        x_rec = glow.reverse(z_s_zero, z_c)
    
    # --- 可视化 ---
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(img.resize((64,64)))
    plt.title("Original")

    plt.subplot(1, 4, 2)
    plt.imshow(z_c[0, 0].cpu().numpy(), cmap='magma')
    plt.title("z_c (Soul) Channel 0")

    plt.subplot(1, 4, 3)
    plt.imshow(z_c[0, 1].cpu().numpy(), cmap='magma')
    plt.title("z_c (Soul) Channel 1")

    plt.subplot(1, 4, 4)
    plt.imshow(x_rec[0].cpu().permute(1,2,0).clamp(0,1).numpy())
    plt.title("Recon (zs=0)")

    plt.savefig('verify_stage1_result.png')
    print("✅ 诊断报告已保存: verify_stage1_result.png")
    plt.show()

if __name__ == "__main__":
    verify()