# --- 1. 核心环境修复 (必须置顶，解决所有版本冲突和导入错误) ---
import sys
import os

# 补丁 A: 修复 huggingface_hub 缺少 cached_download 的问题
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        # 现代版本将其更名为 hf_hub_download，我们手动做一个映射
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
except ImportError:
    pass

import torch

# 补丁 B: 模拟 Intel XPU 接口 (防止 diffusers 硬件探测报错)
if not hasattr(torch, 'xpu'):
    class MockXPU:
        def __init__(self):
            self.device_count = lambda: 0
            self.is_available = lambda: False
        def __getattr__(self, name): return lambda *args, **kwargs: None
    torch.xpu = MockXPU()

# 补丁 C: 强制让 transformers 认为 PyTorch 是可用的 (针对 2.2.2 的版本判定)
import transformers
transformers.utils.import_utils._torch_available = True

# --- 2. 导入核心依赖 (现在的顺序是安全的) ---
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline
import clip

# 导入自定义 Glow 路径
sys.path.append(os.getcwd())
from models.glow_model import SimplifiedGlow

# --- 3. 映射网络 (ConceptBridge) ---
class ConceptBridge(nn.Module):
    def __init__(self, zc_channels=8, img_size=32, clip_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(zc_channels * img_size * img_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, clip_dim) 
        )
    def forward(self, z_c): return self.net(z_c)

# --- 4. 配置 ---
# 虽然在模拟模式下，我们依然尝试调用 MPS (Apple GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASS_MAP = {0: "dog", 1: "elephant", 2: "giraffe", 3: "guitar", 4: "horse", 5: "house", 6: "person"}

# --- 5. 模型加载逻辑 ---
def load_all_models():
    print(f"🖥️  运行设备: {device} | 模式: 终极兼容版 (Rosetta/Intel)")
    
    # 加载 Stable Diffusion 1.5，禁用安全检查器
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float32, # 在模拟模式下，float32 往往比 float16 更稳
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    print("📦 正在加载 Glow & Bridge 权重...")
    glow = SimplifiedGlow().to(device)
    glow.load_state_dict(torch.load('checkpoints/glow_stage1.pth', map_location=device))
    glow.eval()

    bridge = ConceptBridge(zc_channels=8).to(device)
    bridge.load_state_dict(torch.load('checkpoints/bridge_stage2.pth', map_location=device))
    bridge.eval()
    
    return pipe, glow, bridge

# --- 6. 生成函数 ---
def sfda_inference(img_path, pipe, glow, bridge):
    img = Image.open(img_path).convert('RGB')
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # 灵魂提取
        _, z_c, _ = glow.transform_to_noise(x)
        concept_vec = bridge(z_c) 
        
        # 语义识别
        clip_model, _ = clip.load("ViT-B/32", device=device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in CLASS_MAP.values()]).to(device)
        text_features = clip_model.encode_text(text_inputs)
        
        concept_vec /= concept_vec.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (concept_vec @ text_features.T).softmax(dim=-1)
        concept_name = CLASS_MAP[similarity.argmax().item()]
        
    print(f"🎯 识别到灵魂: {concept_name} (置信度: {similarity.max().item():.2f})")
    
    # 灵魂重塑
    image = pipe(
        prompt=f"a realistic high-quality photo of a {concept_name}, ultra detailed, 8k",
        num_inference_steps=20, # 适当减少步数以加快在模拟模式下的速度
        guidance_scale=7.5
    ).images[0]
    
    return image, concept_name

# --- 7. 主程序 ---
if __name__ == "__main__":
    # 路径匹配
    TEST_IMAGE = ""
    target_dir = "data/art_painting"
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                TEST_IMAGE = os.path.join(root, file)
                break
        if TEST_IMAGE: break

    print(f"🚀 SFDA 启动 | 目标: {TEST_IMAGE}")
    
    try:
        sd_pipe, model_glow, model_bridge = load_all_models()
        result, label = sfda_inference(TEST_IMAGE, sd_pipe, model_glow, model_bridge)
        
        os.makedirs('results', exist_ok=True)
        result.save(f"results/final_output_{label}.png")
        print(f"✅ 生成成功！文件: results/final_output_{label}.png")
    except Exception as e:
        print(f"❌ 运行报错: {e}")