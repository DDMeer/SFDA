import torch
from diffusers import StableDiffusionPipeline
from models.glow_model import SimplifiedGlow
from models.mapping_net import StyleMapper

def generate_with_fiber(target_img_path, class_name, glow_ckpt, mapper_ckpt):
    # 1. 加载所有模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    glow = SimplifiedGlow().to(device).eval()
    glow.load_state_dict(torch.load(glow_ckpt))
    
    # 假设 split_dim 在 Squeeze 后是 8 (12个通道取8个)
    mapper = StyleMapper(zs_channels=8, img_size=32).to(device).eval()
    mapper.load_state_dict(torch.load(mapper_ckpt))
    
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
    
    # 2. 提取并映射风格 (Fiber)
    # 假设 target_img 是预处理后的 tensor [1, 3, 64, 64]
    target_img = load_and_preprocess(target_img_path).to(device) 
    with torch.no_grad():
        z, _ = glow.transform_to_noise(target_img)
        z_s = z[:, :8, :, :] # 提取纤维部分
        style_embed = mapper(z_s).unsqueeze(1) # [1, 1, 768]

    # 3. 准备语义 (Base) 并进行注入
    prompt = f"a photo of a {class_name}"
    # 获取原始文本嵌入
    text_inputs = pipe.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt").to(device)
    text_embeds = pipe.text_encoder(text_inputs.input_ids)[0] # [1, 77, 768]
    
    # 关键修改：将 style_embed 替换到文本序列的第一个 padding 位置
    # 这样既不破坏语义，又能让 Cross-Attention 感知到风格
    text_embeds[:, 1:2, :] = style_embed 
    
    # 4. 生成图像
    image = pipe(prompt_embeds=text_embeds).images[0]
    image.save(f"gen_{class_name}.png")

def load_and_preprocess(path):
    # 此处省略标准的图像加载和 Resize(64,64) 代码
    pass