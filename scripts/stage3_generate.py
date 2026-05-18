# --- 1. 硬件与环境补丁 (针对 Mac M2/M3 及现代库版本) ---
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
import argparse

import huggingface_hub
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import torch.distributed
if not hasattr(torch.distributed, 'device_mesh'):
    class Dummy: pass
    torch.distributed.device_mesh = type('Mock', (), {'DeviceMesh': Dummy})

if not hasattr(torch, 'xpu'):
    class MockXPU:
        def __getattr__(self, name): return lambda *args, **kwargs: None
    torch.xpu = MockXPU()

import transformers
transformers.utils.import_utils._torch_available = True

# --- 2. 核心库导入 ---
from diffusers import StableDiffusionImg2ImgPipeline
import clip
sys.path.append(os.getcwd())
from models.glow_model import SimplifiedGlow

# --- 3. 映射网络架构 (严格匹配权重文件) ---
class ConceptBridge(nn.Module):
    def __init__(self, zc_channels=8, img_size=32, clip_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                                       # 0
            nn.Linear(zc_channels * img_size * img_size, 1024), # 1
            nn.BatchNorm1d(1024),                               # 2
            nn.ReLU(),                                          # 3
            nn.Dropout(0.3),                                    # 4
            nn.Linear(1024, 512),                               # 5
            nn.BatchNorm1d(512),                                # 6
            nn.ReLU(),                                          # 7
            nn.Linear(512, clip_dim)                            # 8
        )
    def forward(self, z_c): return self.net(z_c)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASS_MAP = {0: "dog", 1: "elephant", 2: "giraffe", 3: "guitar", 4: "horse", 5: "house", 6: "person"}

# --- 4. 模型加载逻辑 ---
def load_models():
    print(f"🖥️  设备: {device} | 模式: 结构保形重塑 (Img2Img)")
    
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

# --- 5. 推理核心：解耦 -> 语义诊断 -> 重塑 ---
def sfda_inference(img_path, pipe, glow, bridge, forced_label=None):
    raw_img = Image.open(img_path).convert('RGB')
    glow_trans = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    x_glow = glow_trans(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # A. 提取几何底座 z_c (骨架)
        outputs = glow.transform_to_noise(x_glow)
        z_tensors = [t for t in outputs if isinstance(t, torch.Tensor) and t.dim() == 4]
        z_c, z_s = z_tensors[0], z_tensors[1]
        
        # 制造影子底稿 (骨架图)
        z_s_zero = torch.zeros_like(z_s)
        x_rec = glow.reverse(z_c, z_s_zero)
        recon_np = x_rec[0].cpu().permute(1,2,0).clamp(0,1).numpy()
        structure_base = Image.fromarray((recon_np * 255).astype(np.uint8)).resize((512,512))
        
        file_base_name = os.path.basename(img_path).split('.')[0]
        structure_base.save(f"results/base_{file_base_name}_debug.png")
        
        # B. 语义诊断 (识别灵魂标签)
        if forced_label:
            label = forced_label
            print(f"⚠️  手动指定标签为: {label}")
        else:
            # 补齐通道以适配 Bridge 网络
            padding = torch.zeros_like(z_c).to(device)
            z_c_padded = torch.cat([z_c, padding], dim=1)

            concept_vec = bridge(z_c_padded)
            clip_model, _ = clip.load("ViT-B/32", device=device)
            text_tokens = torch.cat([clip.tokenize(f"a photo of a {c}") for c in CLASS_MAP.values()]).to(device)
            text_features = clip_model.encode_text(text_tokens)
            
            # 计算相似度分布
            logits = (concept_vec / concept_vec.norm(dim=-1, keepdim=True) @ 
                     (text_features / text_features.norm(dim=-1, keepdim=True)).T)
            probs = logits.softmax(dim=-1)[0].cpu().numpy()

            # --- 新增：打印详细概率分布 ---
            print("\n📊 语义识别概率分布诊断：")
            for idx, p in enumerate(probs):
                mark = " ⭐" if p == max(probs) else ""
                print(f"  [{idx}] {CLASS_MAP[idx]:<10}: {p:.4f}{mark}")
            
            label = CLASS_MAP[probs.argmax()]
            print(f"🎯 最终提取概念: {label}\n")

    # C. 纤维重塑 (在骨架上编织新纹理)
    # strength=0.55 是保留形状和更新风格的黄金平衡点
    final_image = pipe(
        prompt=f"a realistic high-quality photo of a {label}, cinematic lighting, 8k, highly detailed",
        negative_prompt="cartoon, painting, blurry, lowres, sketch, abstract, cross-eyed",
        image=structure_base,
        strength=0.55, 
        guidance_scale=8.5
    ).images[0]
    
    return final_image, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--force_label', type=str, choices=CLASS_MAP.values())
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    
    try:
        sd, gl, br = load_models()
        res, name = sfda_inference(args.image, sd, gl, br, args.force_label)
        
        out_name = f"results/final_sfda_{os.path.basename(args.image).split('.')[0]}_{name}.png"
        res.save(out_name)
        print(f"✅ 生成成功！\n🖼️  影子骨架: results/base_{os.path.basename(args.image).split('.')[0]}_debug.png\n📸 最终结果: {out_name}")
    except Exception as e:
        import traceback
        traceback.print_exc()