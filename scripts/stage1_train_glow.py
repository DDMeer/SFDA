import torch
from nflows.flows import Glow

# 1. 定义模型 (假设使用 nflows)
def build_glow(img_size=(3, 64, 64)):
    # 这里可以使用 nflows 预定义的 Glow 模板
    # 指定 split 比例，例如 70% 维度作为 z_style (fiber)
    model = Glow(features=12288, hidden_features=512, num_layers=20)
    return model

# 2. 雅可比对齐 Loss (utils/jacobian_loss.py)
def jacobian_alignment_loss(model, x, split_dim):
    x.requires_grad_(True)
    z = model.transform_to_noise(x) # 正向映射
    
    # 随机采样两个向量用于计算 JVP (Jacobian-Vector Product)
    v_s = torch.randn_like(z[:, :split_dim])
    v_c = torch.randn_like(z[:, split_dim:])
    
    # 获取逆变换产生的图像
    x_gen = model.inverse(z)
    
    # 计算切向量 (雅可比与向量的积)
    # 这里的关键是：z_s 的变化方向与 z_c 的变化方向在 X 空间应正交
    grad_s = torch.autograd.grad(x_gen, z, grad_outputs=v_s, create_graph=True)[0]
    grad_c = torch.autograd.grad(x_gen, z, grad_outputs=v_c, create_graph=True)[0]
    
    return torch.dot(grad_s.flatten(), grad_c.flatten())**2