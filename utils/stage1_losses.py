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


def _jvp_double_vjp(fn, primal, tangent):
    """用两次反向模式求导计算 JVP：J·tangent。

    为什么不用 torch.func.jvp：它在 PyTorch 2.2.2 + MPS 上前向可用，但对
    参数 backward 会触发内部断言
    `as_strided_tensorimpl does not work with MPS`。double-VJP 是纯反向模式，
    在 MPS 上可正常回传，且实测与 torch.func.jvp 数值完全一致。

    注意：**不替换 primal**。primal 通常是 z_c = g_c(f; theta)，它本身依赖
    网络参数；若在此处 detach，切空间的求值点就被当成常数，loss 通过「切空间
    所处 latent 位置」回到 forward INN 的梯度路径会被切断，变成 stop-gradient
    近似。实测两者 forward 值相同，但参数梯度的余弦相似度只有约 0.984。
    """
    y = fn(primal)
    u = torch.zeros_like(y, requires_grad=True)
    (jt_u,) = torch.autograd.grad(y, primal, grad_outputs=u, create_graph=True)
    (jv,) = torch.autograd.grad(jt_u, u, grad_outputs=tangent, create_graph=True)
    return jv


def tangent_orthogonality(inverse_fn, z_c, z_s, generator=None, eps=1e-8,
                          detach_latent=False, return_diagnostics=False):
    """L_orth：水平（语义）与垂直（风格）切子空间的正交性。

    采用**逆映射的切空间形式**，这是 research_idea.pdf「特征点的切空间可被
    唯一分解为切于风格纤维的垂直子空间与同构于底流形切空间的水平子空间；
    ……这两个子空间在黎曼度量下的正交可分性」的字面对应：

        h(z_c, z_s) = g^{-1}(z_c, z_s)
        a_c = (∂h/∂z_c) · w      w ~ 单位随机方向 ∈ R^{n_c}   水平切向量
        a_s = (∂h/∂z_s) · v      v ~ 单位随机方向 ∈ R^{n_s}   垂直切向量

        cos    = <a_c, a_s> / (||a_c||·||a_s|| + eps)
        L_orth = mean(cos²)

    说明与限定：
    - 垂直子空间 V = span(∂h/∂z_s 的列) = ker(∂z_c/∂f)，即纤维的切空间；
      水平子空间 H = span(∂h/∂z_c 的列) = ker(∂z_s/∂f)，由平凡化本身诱导的
      平坦联络给出。一般纤维丛的水平子空间需要联络才能确定，这个选择应在
      论文中明写。
    - 度量取特征空间欧氏内积 G = I，首版实现选择；若要让「黎曼」二字承担
      计算内容，需换成拉回度量（如 Fisher）。
    - 用余弦平方而非原始内积平方，是为了**尺度不变**：否则整体缩小 ∂h/∂z_s
      即可降低损失，而不真正把夹角掰正。实测 eps=0 时尺度不变性精确到机器
      精度；默认 eps=1e-8 仅在切向量范数极小时引入约 1e-7 的相对偏差。
    - 这是基于随机方向的**主角度代理**，不是精确主角度或 ||J_c G J_sᵀ||_F。
      精确版本需对两个雅可比块做 QR 后算主角度，代价过高，只适合小批量诊断。
    - 可证明该条件与编码器形式 J_c J_sᵀ = 0 等价（对可逆的 g），但两者作为
      损失函数的梯度与条件数不同。
    - 在零初始化的 INN 上 L_orth ≡ 0 且梯度为零：此时 ActNorm 是对角、混合层
      是置换、耦合层是恒等，整个雅可比为单项矩阵，两组坐标支撑天然不相交。
      它只有在耦合层脱离恒等后才产生信号——这正是它要对抗的漂移。

    Args:
        inverse_fn: 可调用 (z_c, z_s) -> f，通常是 inn.inverse
        z_c: [B, n_c]   z_s: [B, n_s]   应保留计算图（不要在 no_grad 下调用）
        detach_latent: 仅供诊断对照；置 True 会切断经由 latent 求值点的梯度
        return_diagnostics: 额外返回 |cos| 与切向量范数

    Returns:
        标量张量；若 return_diagnostics 则返回 (loss, dict)。
    """
    if detach_latent:
        z_c = z_c.detach().requires_grad_(True)
        z_s = z_s.detach().requires_grad_(True)
    if not (z_c.requires_grad and z_s.requires_grad):
        raise RuntimeError(
            'tangent_orthogonality 需要 z_c/z_s 保留计算图才能求切向量。'
            '请勿在 torch.no_grad() 下调用；纯诊断可传 detach_latent=True。')

    w = torch.randn(z_c.shape, generator=generator, device=z_c.device, dtype=z_c.dtype)
    v = torch.randn(z_s.shape, generator=generator, device=z_s.device, dtype=z_s.dtype)
    w = w / w.norm(dim=1, keepdim=True).clamp_min(eps)
    v = v / v.norm(dim=1, keepdim=True).clamp_min(eps)

    a_c = _jvp_double_vjp(lambda t: inverse_fn(t, z_s), z_c, w)
    a_s = _jvp_double_vjp(lambda t: inverse_fn(z_c, t), z_s, v)

    nc, ns = a_c.norm(dim=1), a_s.norm(dim=1)
    cos = (a_c * a_s).sum(1) / (nc * ns + eps)
    loss = cos.pow(2).mean()
    if not return_diagnostics:
        return loss
    return loss, {'abs_cos': cos.abs().detach(),
                  'norm_a_c': nc.detach(), 'norm_a_s': ns.detach()}
