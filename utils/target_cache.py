"""目标域特征缓存的读取与难/易划分。

划分依据只有源模型输出的预测熵，不使用任何目标域真值。

设计说明：目标域特征最大约 32 MB（sketch），整体常驻内存即可，无需
DataLoader 逐样本搬运。因此这里不继承 Dataset，而是直接持有张量并提供
批迭代器；设备由调用方通过 .to() 显式指定，不写进构造函数。
"""
import os
import sys

import torch

sys.path.append(os.getcwd())
from scripts.cache_features import load_target_cache

CACHE_DIR = 'cache/features'
_TENSOR_FIELDS = ('feature', 'entropy', 'prob', 'logits', 'pseudo_label',
                  'easy_mask', 'hard_mask')


def cache_path(source, target):
    return os.path.join(CACHE_DIR, f'{source}__{target}.pt')


def median_entropy_split(entropy):
    """research_idea.pdf「困难数据 X_h / 简单数据 X_e」的第一版实现。

    tau = median(H)，X_e = {H <= tau}，X_h = {H > tau}。
    返回 (tau, easy_mask, hard_mask)。
    """
    tau = entropy.median()
    easy = entropy <= tau
    return tau, easy, ~easy


class TargetFeatures:
    """缓存好的目标域特征集（张量常驻，默认在 CPU）。

    真值不在此处出现：load_target_cache 默认剥离 eval_only。
    """

    def __init__(self, source, target):
        d = load_target_cache(cache_path(source, target))     # 不含 eval_only
        self.feature = d['feature']
        self.entropy = d['entropy']
        self.prob = d['prob']
        self.logits = d['logits']
        self.pseudo_label = d['pseudo_label']
        self.path = d['path']
        self.meta = d['meta']
        self.source, self.target = source, target
        self.tau, self.easy_mask, self.hard_mask = median_entropy_split(self.entropy)

    def to(self, device):
        """就地把所有张量搬到指定设备，返回 self（便于链式调用）。"""
        for name in _TENSOR_FIELDS:
            setattr(self, name, getattr(self, name).to(device))
        self.tau = self.tau.to(device)
        return self

    @property
    def device(self):
        return self.feature.device

    def __len__(self):
        return self.feature.shape[0]

    @property
    def dim(self):
        return self.feature.shape[1]

    def easy_features(self):
        return self.feature[self.easy_mask]

    def hard_features(self):
        return self.feature[self.hard_mask]

    def iter_batches(self, batch_size, shuffle=True, generator=None, drop_last=False):
        """按批产出 (feature, is_hard, index)，全部已在 self.device 上。

        ⚠️ 排列必须在 **CPU** 上生成后再搬到设备：PyTorch 2.2.2 的 MPS 实现下
           torch.randperm 会忽略 generator 并直接返回恒等排列 [0..n-1]，导致
           shuffle 静默失效。generator 请传 CPU 上的 torch.Generator。
           （实测 MPS 的 randn/rand 正常，只有 randperm 有此缺陷。）
        """
        n = len(self)
        if shuffle:
            order = torch.randperm(n, generator=generator).to(self.feature.device)
        else:
            order = torch.arange(n, device=self.feature.device)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            if drop_last and idx.numel() < batch_size:
                break
            yield self.feature[idx], self.hard_mask[idx], idx

    def summary(self):
        return (f'{self.source} → {self.target} | N={len(self)} dim={self.dim} | '
                f'tau={self.tau:.4f} | easy={int(self.easy_mask.sum())} '
                f'hard={int(self.hard_mask.sum())} | device={self.device}')
