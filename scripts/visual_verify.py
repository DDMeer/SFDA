import torch
import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from models.glow_model import SimplifiedGlow

# 替换成你想要验证的图片路径
IMAGE_PATH = "data/PACS/art_painting/dog/001.jpg" # 请确保路径正确
SAVE_PATH = "verification_results/comparison_res_v2.png"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def verify():
    model = SimplifiedGlow().to(device)
    if os.path.exists('checkpoints/glow_stage1.pth'):
        model.load_state_dict(torch.load('checkpoints/glow_stage1.pth', map_location=device))
    model.eval()

    img = Image.open(IMAGE_PATH).convert('RGB')
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # 编码 (4:8 划分)
        z_s, z_c, _ = model.transform_to_noise(x)
        
        # 原图重构
        x_rec = model.reverse(z_s, z_c)
        
        # 风格增强 (纤维拉伸)
        x_style_up = model.reverse(z_s * 1.5, z_c)
        
        # --- 核心修改：保持内容 + 极低干扰的随机风格 ---
        # Coefficient 0.3 能减少噪声对轮廓的遮盖
        z_s_random = torch.randn_like(z_s) * 0.3 
        x_random_style = model.reverse(z_s_random, z_c)

    res = torch.cat([x, x_rec, x_style_up, x_random_style], dim=0)
    os.makedirs('verification_results', exist_ok=True)
    save_image(res, SAVE_PATH, nrow=4, normalize=True)
    print(f"✅ 质检完成！请查看: {SAVE_PATH}")

if __name__ == "__main__":
    verify()