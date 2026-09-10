"""线性探针与配套工具（仅用于离线评估）。

⚠️ 所有探针都是**评估工具**：输入张量在进入探针前 detach，梯度不会回传到
   INN 或源模型；探针结果不得用于重训 Stage 1 或选择其超参。
   真值标签只在此处的离线诊断中使用。
"""
import torch
import torch.nn as nn


class Standardizer:
    """逐维标准化，统计量只在 train 上拟合。

    ⚠️ 这是为了数值条件化与公平的线性探针优化，**不是**剥离域尺度信息：
       x' = (x - mu)/sigma 是所有样本共享的仿射变换，跨域的相对分布差异
       原样保留。域特有的方差本身可能就是风格信息，不应人为删除。
    """

    def __init__(self, eps=1e-6):
        self.mu = self.sigma = None
        self.eps = eps

    def fit(self, x):
        self.mu = x.mean(0, keepdim=True)
        self.sigma = x.std(0, keepdim=True).clamp_min(self.eps)
        return self

    def __call__(self, x):
        return (x - self.mu) / self.sigma


def _fit_linear(x, y, num_classes, wd, steps, lr, seed):
    """在给定数据上拟合一个线性分类器（L-BFGS + 交叉熵 + L2）。"""
    torch.manual_seed(seed)
    lin = nn.Linear(x.shape[1], num_classes, bias=True).to(x.device)
    opt = torch.optim.LBFGS(lin.parameters(), lr=lr, max_iter=steps,
                            history_size=20, line_search_fn='strong_wolfe')
    ce = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(lin(x), y) + wd * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return lin


@torch.no_grad()
def _acc(lin, x, y):
    return lin(x).argmax(1).eq(y).float().mean().item()


def linear_probe(x_tr, y_tr, x_va, y_va, x_te, y_te, num_classes,
                 weight_decays=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
                 steps=500, lr=1.0, seed=0):
    """多类线性探针，严格的两阶段协议。

    阶段 1（超参搜索）：**只使用 train 与 val**。对每个候选 L2 强度在 train
        上拟合、在 val 上评估，用 val 选出 best_wd。此阶段完全不访问 test。
    阶段 2（最终评估）：用 best_wd 在 **train 上重新拟合一个全新的分类器**，
        然后对 train/val/test **各评估一次**。

    因此可以无歧义地声明：test set was evaluated exactly once after
    hyperparameter selection。此处**不**把 train+val 合并再拟合，保持
    train-only 拟合使比较与泄漏核算更简单。

    返回 dict(best_weight_decay, train, val, test, search)。
    """
    x_tr, x_va, x_te = (t.detach() for t in (x_tr, x_va, x_te))

    # ---- 阶段 1：train -> val，绝不触碰 test ----
    search = []
    best_wd, best_va = None, -1.0
    for wd in weight_decays:
        lin = _fit_linear(x_tr, y_tr, num_classes, wd, steps, lr, seed)
        tr_a, va_a = _acc(lin, x_tr, y_tr), _acc(lin, x_va, y_va)
        search.append({'wd': wd, 'train': tr_a, 'val': va_a})
        if va_a > best_va:
            best_wd, best_va = wd, va_a

    # ---- 阶段 2：用 best_wd 在 train 上重新拟合，各评估一次 ----
    lin = _fit_linear(x_tr, y_tr, num_classes, best_wd, steps, lr, seed)
    return dict(best_weight_decay=best_wd,
                train=_acc(lin, x_tr, y_tr),
                val=_acc(lin, x_va, y_va),
                test=_acc(lin, x_te, y_te),
                search=search)


def probe_representation(name, splits, num_classes, seed=0, **kw):
    """对一种表征跑完整探针流程（train 上拟合标准化 → 选 wd → 评估 test 一次）。

    splits: {'train':(x,y), 'val':(x,y), 'test':(x,y)}。
    探针在 CPU 上运行：数据规模小，且更易保证可复现。
    """
    splits = {k: (v[0].detach().cpu().float(), v[1].cpu()) for k, v in splits.items()}
    st = Standardizer().fit(splits['train'][0])          # 统计量只来自 train
    xs = {k: (st(v[0]), v[1]) for k, v in splits.items()}
    r = linear_probe(xs['train'][0], xs['train'][1],
                     xs['val'][0], xs['val'][1],
                     xs['test'][0], xs['test'][1],
                     num_classes, seed=seed, **kw)
    r.update(name=name, dim=splits['train'][0].shape[1],
             n_train=len(splits['train'][1]), n_val=len(splits['val'][1]),
             n_test=len(splits['test'][1]))
    return r


class FrozenProbe:
    """已拟合并冻结的探针：Standardizer + 线性分类器 + 选定的 L2 强度。

    用于 robustness 分析——只重采样 test 子集，复用**同一个**冻结探针评估，
    从而单独度量 test 采样方差，而不把训练随机性与超参选择混进来。
    """

    def __init__(self, standardizer, linear, best_wd):
        self.st, self.lin, self.best_wd = standardizer, linear, best_wd
        self.lin.eval()                       # 真正冻结，防误用
        for prm in self.lin.parameters():
            prm.requires_grad_(False)
        # Standardizer 在 train 上 fit 之后即固定，无需额外处理

    @torch.no_grad()
    def accuracy(self, x, y):
        x = self.st(x.detach().cpu().float())
        return self.lin(x).argmax(1).eq(y.cpu()).float().mean().item()


def fit_probe(name, splits, num_classes, seed=0,
              weight_decays=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), steps=500, lr=1.0):
    """拟合探针并**同时返回冻结对象**，供 robustness 复用。

    协议与 linear_probe 相同：超参搜索只用 train/val，test 在 best_wd 固定后
    只评估一次。
    """
    sp = {k: (v[0].detach().cpu().float(), v[1].cpu()) for k, v in splits.items()}
    st = Standardizer().fit(sp['train'][0])              # 统计量只来自 train
    xs = {k: (st(v[0]), v[1]) for k, v in sp.items()}

    best_wd, best_va, search = None, -1.0, []
    for wd in weight_decays:                              # 阶段 1：不触碰 test
        lin = _fit_linear(xs['train'][0], xs['train'][1], num_classes, wd, steps, lr, seed)
        tr_a = _acc(lin, xs['train'][0], xs['train'][1])
        va_a = _acc(lin, xs['val'][0], xs['val'][1])
        search.append({'wd': wd, 'train': tr_a, 'val': va_a})
        if va_a > best_va:
            best_wd, best_va = wd, va_a

    lin = _fit_linear(xs['train'][0], xs['train'][1], num_classes, best_wd, steps, lr, seed)
    res = dict(name=name, dim=sp['train'][0].shape[1], best_weight_decay=best_wd,
               train=_acc(lin, xs['train'][0], xs['train'][1]),
               val=_acc(lin, xs['val'][0], xs['val'][1]),
               test=_acc(lin, xs['test'][0], xs['test'][1]),   # 唯一一次 test 评估
               n_train=len(sp['train'][1]), n_val=len(sp['val'][1]),
               n_test=len(sp['test'][1]), search=search)
    return res, FrozenProbe(st, lin, best_wd)
