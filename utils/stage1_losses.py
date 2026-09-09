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


def semantic_consistency(logits_f, logits_fc, temperature=1.0):
    """L_sem：去掉风格纤维后，冻结源分类器的判定应保持不变。

        p     = softmax(C(f))          冻结的 teacher，detach
        f_c   = g^{-1}(z_c, 0)
        q     = softmax(C(f_c))
        L_sem = (1/B) * sum_b KL(p_b || q_b)          KL 对 7 个类别求和

    这是 research_idea.pdf 中「跨域的风格变换在几何上仅表现为特征点沿着
    垂直纤维方向的滑移，而其在底流形上的投影点保持不动」的计算形式：
    把 z_s 置零是沿纤维的一次移动，底流形上的语义投影应当不变。

    ⚠️ 它**不是**用伪标签做类别监督。即使源模型在目标域上预测错误，这个
       一致性约束依然成立——要求的是「保持源模型原本的语义判定」，而非
       「预测正确的类别」。因此不需要、也不使用任何目标域真值或伪标签。

    ⚠️ L_sem **不能**证明 z_s 不含语义信息。它只强制 z_c 足以在去掉 z_s
       之后保持源分类器的判定（充分性），并不约束 z_s 的类别无关性。
       单凭 L_sem 不构成解耦证据。

    Args:
        logits_f:  [B, K]  C(f)，teacher；函数内部会 detach
        logits_fc: [B, K]  C(g^{-1}(z_c, 0))，梯度经此回到 INN
        temperature: 首版固定为 1.0

    Returns:
        标量张量。
    """
    log_p = torch.log_softmax(logits_f.detach() / temperature, dim=1)
    log_q = torch.log_softmax(logits_fc / temperature, dim=1)
    p = log_p.exp()
    # KL(p||q) = sum_k p_k (log p_k - log q_k)，对类别求和、对样本取均值
    return (p * (log_p - log_q)).sum(1).mean()


def style_removal_ratio(f, f_c):
    """诊断量 r_style = ||f - f_c|| / ||f||，逐样本后取均值。

    若 L_sem 下降的同时 r_style → 0，说明 INN 可能退化为「把几乎所有信息
    都塞进 z_c、z_s 不起作用」的平凡解。健康状态应是语义一致性上升而
    r_style 仍显著非零。
    """
    return ((f - f_c).norm(dim=1) / f.norm(dim=1).clamp_min(1e-12)).mean()
