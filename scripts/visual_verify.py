import torch
import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from models.glow_model import SimplifiedGlow

# --- 修改这里：确保路径指向一张真实存在的图片 ---
IMAGE_PATH = "data/art_painting/0/0.jpg" 
SAVE_PATH = "verification_results/comparison_res.png"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def verify():
    model = SimplifiedGlow().to(device)
    if os.path.exists('checkpoints/glow_stage1.pth'):
        model.load_state_dict(torch.load('checkpoints/glow_stage1.pth', map_location=device))
    model.eval()

    # 加载图片
    img = Image.open(IMAGE_PATH).convert('RGB')
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # 1. 编码到隐空间 (6:6 划分)
        z_s, z_c, _ = model.transform_to_noise(x)
        
        # 2. 原图重构
        x_rec = model.reverse(z_s, z_c)
        
        # 3. 风格增强 (纤维拉伸)
        x_style_up = model.reverse(z_s * 1.5, z_c)
        
        # 4. 保持语义 + 随机风格 (核心测试)
        # 使用 0.6 的温度系数，防止随机值太大导致乱码
        z_s_random = torch.randn_like(z_s) * 0.6 
        x_random_style = model.reverse(z_s_random, z_c)

    # 合并保存：原图 | 重构 | 风格增强 | 随机风格(内容保持)
    res = torch.cat([x, x_rec, x_style_up, x_random_style], dim=0)
    os.makedirs('verification_results', exist_ok=True)
    save_image(res, SAVE_PATH, nrow=4, normalize=True)
    print(f"✅ 验证完成！结果已保存至: {SAVE_PATH}")

if __name__ == "__main__":
    verify()