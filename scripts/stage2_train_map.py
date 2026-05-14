import torch
import torch.nn as nn
from models.glow_model import build_glow
from models.mapping_net import StyleMapper
from transformers import CLIPVisionModel, CLIPImageProcessor

def train_mapping_stage(source_loader):
    # 1. 初始化模型
    glow = build_glow().eval() # 冻结 Glow
    mapper = StyleMapper(zs_dim=8192).train() # 只训练这个！
    
    # 加载预训练的 CLIP 作为“老师”
    clip_teacher = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    optimizer = torch.optim.Adam(mapper.parameters(), lr=1e-4)
    criterion = nn.MSELoss() # 简单的对齐损失

    for images, _ in source_loader:
        # A. 提取物理纤维坐标 z_s (来自 Glow)
        with torch.no_grad():
            z = glow.transform_to_noise(images)
            z_s = z[:, :split_dim].flatten(1) # 展平 z_s
        
        # B. 提取语义/风格参考向量 v_clip (来自 CLIP)
        with torch.no_grad():
            # 这里的 images 可能需要 resize/normalize
            inputs = clip_processor(images, return_tensors="pt")
            v_clip = clip_teacher(**inputs).last_hidden_state[:, 0, :] # 取 [CLS] token
            
        # C. 映射与对齐
        v_mapped = mapper(z_s)
        
        # D. 计算 Loss
        # 1. 基础对齐损耗：让映射后的向量在数学上接近 CLIP 的表征
        loss_align = criterion(v_mapped, v_clip)
        
        # 2. (可选) 正交正则化：确保不同 z_s 映射出的 v_mapped 具有区分度
        # loss_ortho = ... 
        
        loss = loss_align
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # 保存这个“翻译官”
    torch.save(mapper.state_dict(), "checkpoints/mapper_final.pth")