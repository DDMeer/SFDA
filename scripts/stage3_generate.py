from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# 提取目标域一张图的风格
z = glow_model(target_img)
z_s = z[:, :split_dim]
style_embed = mapping_net(z_s) # 得到纤维坐标

# 准备底座 (文本概念)
prompt = "a photo of a [Classname]"
text_inputs = pipe.tokenizer(prompt, return_tensors="pt")
text_embeds = pipe.text_encoder(text_inputs.input_ids)[0]

# 核心：拼接风格与语义 (或者使用更高级的 Cross-Attention 注入)
# 这里简化为将 style_embed 作为一个特殊 token 拼在 text_embeds 前面
combined_cond = torch.cat([style_embed.unsqueeze(1), text_embeds], dim=1)

# 生成图片
image = pipe(prompt_embeds=combined_cond).images[0]