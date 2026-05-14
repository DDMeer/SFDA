import torch

def compute_jacobian_alignment(model, x, split_dim):
    """
    强制 z_s (纤维) 和 z_c (底座) 在图像空间的变化方向正交。
    """
    x.requires_grad_(True)
    z, _ = model.transform_to_noise(x)
    
    # 拆分隐变量
    z_s = z[:, :split_dim, :, :]
    z_c = z[:, split_dim:, :, :]
    
    # 随机采样切向量 v
    v_s = torch.randn_like(z_s)
    v_c = torch.randn_like(z_c)
    
    # 使用逆变换计算生成图像
    # 我们希望针对 z_s 的微扰产生的图像变化与 z_c 产生的变化正交
    x_gen = model.reverse(z)
    
    # 计算 JVP (Jacobian-Vector Product)
    # grad_s 代表沿着纤维方向的图像切向量
    grad_s = torch.autograd.grad(
        outputs=x_gen, 
        inputs=z, 
        grad_outputs=torch.cat([v_s, torch.zeros_like(z_c)], dim=1),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # grad_c 代表沿着底座语义方向的图像切向量
    grad_c = torch.autograd.grad(
        outputs=x_gen, 
        inputs=z, 
        grad_outputs=torch.cat([torch.zeros_like(z_s), v_c], dim=1),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # 损失函数：最小化两个切向量的点积平方
    # 公式：L = (g_s · g_c)^2
    cos_sim = (grad_s * grad_c).sum() 
    loss_ja = cos_sim ** 2
    
    return loss_ja