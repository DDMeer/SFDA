"""特征空间可逆网络（research_idea.pdf「参数共享的双流可逆神经网络」）。

在源模型提取的 2048 维特征上做无损拆解：
    f [B, 2048]  --forward-->  z_c [B, n_c] (类别特征 / 底流形语义)
                               z_s [B, n_s] (风格特征 / 纤维特异性)
    (z_c, z_s)   --inverse-->  f [B, 2048]

inverse 接受任意 z_s：L_sem 传零向量，swap 实验传另一样本的 z_s，同一接口。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActNorm1d(nn.Module):
    """逐维仿射；首个 batch 数据依赖初始化为零均值单位方差。"""

    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        # buffer 的 Python 镜像：避免每次 forward 都 .item() 触发设备同步
        # （实测 MPS 上每层一次同步会让 forward 耗时翻倍）
        self._init_done = None

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._init_done = None          # 强制下次 forward 重新读取 buffer

    @torch.no_grad()
    def _init(self, x):
        m, s = x.mean(0), x.std(0).clamp_min(1e-6)
        self.bias.copy_(-m / s)
        self.log_scale.copy_(-s.log())
        self.initialized.fill_(1)
        self._init_done = True

    def forward(self, x):
        if self._init_done is None:                    # 每个实例只同步一次
            self._init_done = bool(self.initialized.item())
        if not self._init_done:
            if not self.training:
                raise RuntimeError(
                    'ActNorm1d 尚未初始化却在 eval 模式下被调用。'
                    '未初始化时该层等价于恒等映射，会静默返回未归一化的结果。'
                    '请先在 train 模式下用一个真实数据 batch 走一次 forward，'
                    '或加载一个已初始化的 state_dict。')
            self._init(x)
        y = x * self.log_scale.exp() + self.bias
        logdet = self.log_scale.sum().expand(x.shape[0])
        return y, logdet

    def inverse(self, y):
        return (y - self.bias) * (-self.log_scale).exp()


class InvLinearLU(nn.Module):
    """LU 参数化的可逆线性层。

    W = P · L · (U + diag(s))，logdet = Σ log|s| —— O(d) 而非 O(d³)，
    且 s 恒不为零，不会退化为奇异（旧的自由矩阵 + slogdet 有此风险）。
    逆变换用两次三角求解，O(d²)，不显式求逆。
    """

    def __init__(self, dim):
        super().__init__()
        w = torch.linalg.qr(torch.randn(dim, dim))[0]
        P, L, U = torch.linalg.lu(w)
        s = U.diagonal().clone()
        self.register_buffer('P', P)
        self.register_buffer('eye', torch.eye(dim))
        self.register_buffer('l_mask', torch.tril(torch.ones(dim, dim), -1))
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(torch.triu(U, 1))
        self.sign_s = nn.Parameter(s.sign(), requires_grad=False)
        self.log_s = nn.Parameter(s.abs().clamp_min(1e-6).log())

    def _components(self):
        L = self.L * self.l_mask + self.eye                       # 单位下三角
        U = self.U * self.l_mask.T + torch.diag(self.sign_s * self.log_s.exp())
        return L, U

    def forward(self, x):
        L, U = self._components()
        W = self.P @ L @ U
        return F.linear(x, W), self.log_s.sum().expand(x.shape[0])

    def inverse(self, y):
        # y = x Wᵀ  ⟹  xᵀ = U⁻¹ L⁻¹ Pᵀ yᵀ
        L, U = self._components()
        v = (self.P.T @ y.T).contiguous()
        w = torch.linalg.solve_triangular(L, v, upper=False, unitriangular=True)
        return torch.linalg.solve_triangular(U, w, upper=True).T


class AffineCoupling1d(nn.Module):
    """仿射耦合：前半不变，后半按前半预测的尺度/平移变换。"""

    def __init__(self, dim, hidden=1024):
        super().__init__()
        self.d1 = dim // 2
        self.d2 = dim - self.d1
        self.net = nn.Sequential(
            nn.Linear(self.d1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * self.d2),
        )
        nn.init.zeros_(self.net[-1].weight)          # 零初始化 -> 初始为恒等映射
        nn.init.zeros_(self.net[-1].bias)

    def _st(self, x1):
        s, t = self.net(x1).chunk(2, dim=1)
        return torch.tanh(s), t                      # tanh 限幅，允许放大与缩小

    def forward(self, x):
        x1, x2 = x[:, :self.d1], x[:, self.d1:]
        s, t = self._st(x1)
        return torch.cat([x1, x2 * s.exp() + t], 1), s.sum(1)

    def inverse(self, y):
        y1, y2 = y[:, :self.d1], y[:, self.d1:]
        s, t = self._st(y1)
        return torch.cat([y1, (y2 - t) * (-s).exp()], 1)


class FeatureINN(nn.Module):
    """参数共享的特征空间 INN。X_e 与 X_h 由同一实例处理（即「参数共享双流」）。

    mixing 默认 'perm'（固定随机置换）而非 'lu'，理由是实测：
      2048 维正交矩阵 W 本身良态 (cond=1.0)，但其 LU 因子 cond(U)≈4.6e3，
      float32 下三角求解的相对误差 ≈ cond(U)·eps ≈ 5e-4，往返误差达 1e-3；
      而 MPS 不支持 float64，无法用高精度求解绕开。
      置换的逆是精确的索引操作，往返误差 ~3e-6，且参数量少 33M。
    代价是失去可学习的通道混合，需要靠更多 block 补偿。mixing='lu' 仍可选。
    """

    def __init__(self, dim=2048, n_c=1536, n_blocks=4, hidden=1024, mixing='perm'):
        super().__init__()
        assert 0 < n_c < dim, 'n_c 必须在 (0, dim) 内'
        self.dim, self.n_c, self.n_s = dim, n_c, dim - n_c
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            mix = InvLinearLU(dim) if mixing == 'lu' else _FixedPerm(dim)
            self.blocks.append(nn.ModuleList([ActNorm1d(dim), mix,
                                              AffineCoupling1d(dim, hidden)]))

    def split(self, z):
        return z[:, :self.n_c], z[:, self.n_c:]

    def merge(self, z_c, z_s):
        return torch.cat([z_c, z_s], dim=1)

    def forward(self, f):
        logdet = f.new_zeros(f.shape[0])
        z = f
        for act, mix, coup in self.blocks:
            z, ld = act(z);  logdet = logdet + ld
            z, ld = mix(z);  logdet = logdet + ld
            z, ld = coup(z); logdet = logdet + ld
        z_c, z_s = self.split(z)
        return z_c, z_s, logdet

    def inverse(self, z_c, z_s):
        z = self.merge(z_c, z_s)
        for act, mix, coup in reversed(self.blocks):
            z = coup.inverse(z)
            z = mix.inverse(z)
            z = act.inverse(z)
        return z


class _FixedPerm(nn.Module):
    """固定随机置换（logdet=0），LU 层的廉价替代。"""

    def __init__(self, dim):
        super().__init__()
        p = torch.randperm(dim)
        self.register_buffer('perm', p)
        self.register_buffer('inv_perm', torch.argsort(p))

    def forward(self, x):
        return x[:, self.perm], x.new_zeros(x.shape[0])

    def inverse(self, y):
        return y[:, self.inv_perm]
