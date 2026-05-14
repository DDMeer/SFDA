import torch

def compute_jacobian_alignment(model, x, split_dim=8):
    """
    通过随机投影法最小化风格轴与语义轴的切向量相关性
    """
    x.requires_grad_(True)
    z, _ = model.transform_to_noise(x)
    
    # 定义采样向量
    v_s = torch.randn_like(z[:, :split_dim, :, :])
    v_c = torch.randn_like(z[:, split_dim:, :, :])
    
    # 构造两个扰动路径
    z_s_perturbed = torch.cat([z[:, :split_dim, :, :] + v_s * 1e-3, z[:, split_dim:, :, :]], dim=1)
    z_c_perturbed = torch.cat([z[:, :split_dim, :, :], z[:, split_dim:, :, :] + v_c * 1e-3], dim=1)
    
    # 逆变换还原到图像空间
    x_s = model.reverse(z_s_perturbed)
    x_c = model.reverse(z_c_perturbed)
    
    # 计算图像空间的变化量 (近似雅可比切向量)
    diff_s = (x_s - x).flatten(1)
    diff_c = (x_c - x).flatten(1)
    
    # 最小化余弦相似度（或点积）
    loss_ja = torch.mean(torch.abs(torch.cosine_similarity(diff_s, diff_c)))
    
    return loss_ja