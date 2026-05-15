import torch
import os
from torchvision import transforms, utils
from PIL import Image
from models.glow_model import SimplifiedGlow

# --- 配置 ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT = "checkpoints/glow_stage1.pth"
# 这里的路径要指向你现在 data 文件夹下真实存在的图片
# 根据你的截图，0 文件夹下有 0.jpg
IMAGE_PATH = "data/cartoon/1/2437.jpg" # 换成你文件夹里真实存在的图
OUTPUT_DIR = "verification_results"
SPLIT_DIM = 8

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 1. 加载模型 ---
model = SimplifiedGlow(num_layers=12).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# --- 2. 预处理图像 ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

img = Image.open(IMAGE_PATH).convert('RGB')
x = transform(img).unsqueeze(0).to(DEVICE) # [1, 3, 64, 64]

# --- 3. 执行验证 ---
with torch.no_grad():
    # A. 正向映射到隐空间
    z, _ = model.transform_to_noise(x)
    
    # B. 任务一：完美重构验证 (Reconstruction)
    x_rec = model.reverse(z)
    
    # C. 任务二：纤维操纵 (Fiber Manipulation)
    # 我们保持 z_c (后4通道) 不动，给 z_s (前8通道) 增加一点扰动
    z_manipulated = z.clone()
    # 比如：将风格强度放大 1.5 倍，或者加入随机噪声
    z_manipulated[:, :SPLIT_DIM, :, :] *= 1.5 
    x_style_boosted = model.reverse(z_manipulated)
    
    # 换一种：随机采样一个风格，配合原来的语义
    z_random_style = torch.randn_like(z)
    z_random_style[:, SPLIT_DIM:, :, :] = z[:, SPLIT_DIM:, :, :] # 植入原图的语义
    x_mixed = model.reverse(z_random_style)

# --- 4. 保存结果对比 ---
def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)

comparison = torch.cat([
    denormalize(x), 
    denormalize(x_rec), 
    denormalize(x_style_boosted),
    denormalize(x_mixed)
], dim=0)

utils.save_image(comparison, f"{OUTPUT_DIR}/comparison_res.png", nrow=4)
print(f"✅ 验证完成！请查看 {OUTPUT_DIR}/comparison_res.png")
print(f"顺序：原图 | 重构图 | 风格增强图 | 语义保持-随机风格图")