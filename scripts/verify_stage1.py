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
        # --- 健壮的解包逻辑 ---
        outputs = glow.transform_to_noise(x)
        
        # 寻找 4 维的张量作为 z_c 和 z_s
        z_tensors = [t for t in outputs if isinstance(t, torch.Tensor) and t.dim() == 4]
        
        if len(z_tensors) < 2:
            # 兼容另一种实现：只返回了一个合并后的 z 和一个 log_det
            z_full = z_tensors[0]
            z_c = z_full[:, :8, :, :] # 假设前 8 通道是 z_c
            z_s = z_full[:, 8:, :, :] # 假设后 8 通道是 z_s
        else:
            # 假设第一个 4D 张量是 z_c，第二个是 z_s
            z_c, z_s = z_tensors[0], z_tensors[1]

        print(f"📊 提取成功: z_c 维度 {z_c.shape}, z_s 维度 {z_s.shape}")

        # --- 执行逆向重构 (zs 归零) ---
        z_s_zero = torch.zeros_like(z_s)
        try:
            x_rec = glow.reverse(z_c, z_s_zero)
        except Exception as e:
            # 如果 reverse 函数的参数顺序反了，尝试调换
            print("🔄 尝试调换 reverse 参数顺序...")
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