import torch
import os
import sys
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as T

# --- 1. 终极环境补丁 ---
import huggingface_hub
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

if not hasattr(torch, 'xpu'):
    class MockXPU:
        def __getattr__(self, name): return lambda *args, **kwargs: None
    torch.xpu = MockXPU()

# --- 2. 导入依赖 ---
from diffusers import StableDiffusionImg2ImgPipeline
import clip
sys.path.append(os.getcwd())
from models.glow_model import SimplifiedGlow

# --- 3. 映射网络 ---
class ConceptBridge(nn.Module):
    def __init__(self, zc_channels=8, img_size=32, clip_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(zc_channels * img_size * img_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, clip_dim) 
        )
    def forward(self, z_c): return self.net(z_c)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASS_MAP = {0: "dog", 1: "elephant", 2: "giraffe", 3: "guitar", 4: "horse", 5: "house", 6: "person"}

# --- 4. 模型加载 ---
def load_models():
    print(f"🖥️  设备: {device} | 正在开启 [保形重构] 模式")
    
    # 使用 Img2Img 管道，这是保形的关键
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    glow = SimplifiedGlow().to(device)
    glow.load_state_dict(torch.load('checkpoints/glow_stage1.pth', map_location=device))
    glow.eval()

    bridge = ConceptBridge(zc_channels=8).to(device)
    bridge.load_state_dict(torch.load('checkpoints/bridge_stage2.pth', map_location=device))
    bridge.eval()
    
    return pipe, glow, bridge

# --- 5. 核心逻辑 ---
def sfda_reconstruct(img_path, pipe, glow, bridge):
    raw_img = Image.open(img_path).convert('RGB')
    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()]) # SD 需要 512x512
    glow_transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    
    x_glow = glow_transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # A. 剥离纤维，提取底座灵魂
        outputs = glow.transform_to_noise(x_glow)
        z_tensors = [t for t in outputs if isinstance(t, torch.Tensor) and t.dim() == 4]
        z_c, z_s = z_tensors[0], z_tensors[1]
        
        # B. 生成“影子底稿” (保形关键)
        z_s_zero = torch.zeros_like(z_s)
        x_rec = glow.reverse(z_c, z_s_zero)
        
        # 将影子图转为 PIL 格式作为 SD 的引导
        recon_base = x_rec[0].cpu().permute(1,2,0).clamp(0,1).numpy()
        recon_base_img = Image.fromarray((recon_base * 255).astype(np.uint8)).resize((512,512))
        recon_base_img.save("results/debug_base_structure.png") # 供你检查底稿
        
        # C. 语义识别
        concept_vec = bridge(z_c)
        clip_model, _ = clip.load("ViT-B/32", device=device)
        text_tokens = torch.cat([clip.tokenize(f"a photo of a {c}") for c in CLASS_MAP.values()]).to(device)
        text_features = clip_model.encode_text(text_tokens)
        similarity = (concept_vec / concept_vec.norm(dim=-1, keepdim=True) @ (text_features / text_features.norm(dim=-1, keepdim=True)).T).softmax(dim=-1)
        label = CLASS_MAP[similarity.argmax().item()]

    print(f"🎯 识别概念: {label} | 影子底稿已生成")

    # D. 结合影子底稿和语义标签进行重绘
    # strength=0.6 表示保留 60% 的影子轮廓，40% 由 SD 自由发挥写实纹理
    final_image = pipe(
        prompt=f"a realistic high-quality photo of a {label}, highly detailed, national geographic style",
        image=recon_base_img,
        strength=0.6, 
        guidance_scale=8.0
    ).images[0]
    
    return final_image, label

if __name__ == "__main__":
    TEST_PATH = "data/art_painting/0/63.jpg"
    os.makedirs('results', exist_ok=True)
    
    sd, gl, br = load_models()
    res, name = sfda_reconstruct(TEST_PATH, sd, gl, br)
    
    res.save(f"results/final_recon_{name}.png")
    print(f"✅ 保形重构完成！请对比 results/debug_base_structure.png 和 final_recon_{name}.png")