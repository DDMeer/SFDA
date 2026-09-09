"""Stage 1 损失项。

目前只包含 L_flow。L_sem 与 L_orth 尚未实现，不放占位实现。
"""
import torch


def flow_nll(z_c, z_s, logdet, dim=None):
    """标准正态先验下的负对数似然，按维度归一化。

        z       = concat(z_c, z_s)
        L_flow  = mean_b( 0.5 * ||z_b||^2 - logdet_b ) / dim

    高斯常数项 0.5 * dim * log(2*pi) 被省略：它与模型参数无关，对梯度没有
    任何贡献，只会给 loss 加一个固定偏移（dim=2048 时约 0.9189/维）。因此
    这里的数值不是可比的 nats/dim，只用于优化与相对比较。

    ⚠️ L_flow 纯粹是密度建模目标。它对完整的 z=[z_c, z_s] 做联合标准高斯
       建模，**不会**赋予 z_c 类别语义、也不会赋予 z_s 风格语义。角色的
       指派要靠后续的 L_sem 与几何约束。L_flow 下降本身不构成解耦证据。

    Args:
        z_c: [B, n_c]
        z_s: [B, n_s]
        logdet: [B]  前向变换的 log|det J|，逐样本
        dim: 归一化用的维度；默认取 n_c + n_s

    Returns:
        标量张量。梯度可回传至 z_c、z_s 与 logdet。
    """
    z = torch.cat([z_c, z_s], dim=1)
    if dim is None:
        dim = z.shape[1]
    quad = 0.5 * z.pow(2).sum(1)          # [B]
    return (quad - logdet).mean() / dim


def flow_nll_terms(z_c, z_s, logdet, dim=None):
    """诊断用：分别返回二次项与 logdet 项（均已按维度归一化）。

    用于检查 L_flow 的下降是否健康——若靠 logdet 极端增大、同时隐变量
    数值爆炸来换取，则不是健康优化。
    """
    z = torch.cat([z_c, z_s], dim=1)
    if dim is None:
        dim = z.shape[1]
    quad = (0.5 * z.pow(2).sum(1)).mean() / dim
    ld = logdet.mean() / dim
    return quad, ld
