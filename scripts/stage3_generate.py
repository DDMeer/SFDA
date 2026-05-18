import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import sys
import torch.nn.functional as F

sys.path.append(os.getcwd())
from models.glow_model import SimplifiedGlow
from scripts.stage2_train_bridge import ConceptBridge

# --- 1. 配置与类别映射 ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# PACS 数据集标准的类别顺序
CLASS_MAP = {
    0: "dog", 
    1: "elephant", 
    2: "giraffe", 
    3: "guitar", 
    4: "horse", 
    5: "house", 
    6: "person"
}

# --- 2. 加载模型函数 ---
def load_models():
    print("📦 正在加载 Stable Diffusion 1.5 (纤维生成器)...")
    # 使用 float16 减少显存占用
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16
    ).to(device)
    
    print("📦 正在加载 Glow (灵魂剥离器) & Bridge (语义翻译官)...")
    glow = SimplifiedGlow().to(device)
    glow.load_state_dict(torch.load('checkpoints/glow_stage1.pth', map_location=device))
    glow.eval()

    bridge = ConceptBridge(zc_channels=8).to(device)
    bridge.load_state_dict(torch.load('checkpoints/bridge_stage2.pth', map_location=device))
    bridge.eval()
    
    return pipe, glow, bridge

# --- 3. 核心推理逻辑：跨域生成 ---
def sfda_inference(img_path, pipe, glow, bridge, target_style="a realistic photo"):
    from torchvision import transforms
    
    # A. 图像预处理
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # B. 纤维丛分解：提取底座概念 z_c
        _, z_c, _ = glow.transform_to_noise(x)
        
        # C. 语义对齐：将 z_c 映射到语义向量
        concept_vec = bridge(z_c) # [1, 512]
        
        # D. 概念识别：自动判断图片里是什么
        # 这里我们用映射出的向量去匹配最接近的类别
        # 这一步体现了“底座即概念”的逻辑
        import clip
        text_inputs = torch.cat([clip.tokenize(f"a {c}") for c in CLASS_MAP.values()]).to(device)
        clip_model, _ = clip.load("ViT-B/32", device=device)
        text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        concept_vec /= concept_vec.norm(dim=-1, keepdim=True)
        
        similarity = (concept_vec @ text_features.T).softmax(dim=-1)
        class_idx = similarity.argmax().item()
        concept_name = CLASS_MAP[class_idx]
        
    print(f"🎯 识别到概念底座为: {concept_name} (置信度: {similarity.max().item():.2f})")
    
    # E. 灵魂重塑：使用 SD 注入新的“写实纤维”
    final_prompt = f"{target_style} of a {concept_name}, highly detailed, national geographic style"
    print(f"✨ 正在生成: {final_prompt}")
    
    image = pipe(
        prompt=final_prompt,
        num_inference_steps=30,
        guidance_scale=8.0
    ).images[0]
    
    return image, concept_name

# --- 4. 运行 ---
if __name__ == "__main__":
    # 请确保路径下有一张测试图
    TEST_IMAGE = "data/PACS/art_painting/dog/001.jpg" 
    
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ 找不到测试图: {TEST_IMAGE}")
        sys.exit()

    pipe, glow, bridge = load_models()
    
    result_img, name = sfda_inference(TEST_IMAGE, pipe, glow, bridge)
    
    os.makedirs('results', exist_ok=True)
    result_img.save(f"results/translated_{name}.png")
    print(f"✅ 转换完成！结果已保存至 results/translated_{name}.png")