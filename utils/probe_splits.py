"""探针用的数据划分构造（防泄漏）。

类别探针：直接使用 splits/ 下已提交的确定性 80/10/10 划分。
域探针：在**每个 split 内部各自**做类别×域均衡。

均衡的性质（务必准确表述）：
    对每个类别 c，  N(cartoon,c) = N(photo,c) = N(sketch,c)
因此三个域的**类别分布完全一致**，类别与域在该采样集中去相关。
这**不是**说每个域内 7 个类别数量相等——它们并不相等。
"""
import os
import sys

import torch

sys.path.append(os.getcwd())
from models.source_model import PACS_CLASSES
from scripts.prepare_pacs import load_split
from scripts.cache_features import load_target_cache
from utils.target_cache import cache_path

N_CLASSES = len(PACS_CLASSES)
PROBE_DOMAINS = ('cartoon', 'photo', 'sketch')   # 主域探针不含 art_painting


def load_domain(source, domain):
    """返回 (feature[N,2048], gt_label[N], rel_path list)。真值仅供离线诊断。"""
    d = load_target_cache(cache_path(source, domain), include_eval_only=True)
    return d['feature'], d['eval_only']['label'], d['path']


def split_index(domain, split, paths):
    """把 splits/{domain}_{split}.txt 的路径映射到缓存内的行索引。"""
    want = {rel for rel, _ in load_split(domain, split)}
    pos = {p: i for i, p in enumerate(paths)}
    missing = want - pos.keys()
    if missing:
        raise KeyError(f'{domain}/{split}: 缓存中缺少 {len(missing)} 个划分内的样本')
    return torch.tensor(sorted(pos[p] for p in want), dtype=torch.long)


def balanced_domain_indices(per_domain, split, seed=0):
    """在单个 split 内部做类别×域均衡。

    per_domain: {domain: (labels_of_this_split[LongTensor], global_idx[LongTensor])}
    返回 {domain: 选中的 global_idx}，满足对每个类别 c 三域数量相同。
    """
    counts = {d: torch.bincount(lab, minlength=N_CLASSES)
              for d, (lab, _) in per_domain.items()}
    per_class = [min(int(counts[d][c]) for d in per_domain) for c in range(N_CLASSES)]
    out = {}
    for d, (lab, gidx) in per_domain.items():
        keep = []
        for c in range(N_CLASSES):
            pool = gidx[lab == c]
            g = torch.Generator().manual_seed(seed * 1000 + c)
            perm = torch.randperm(pool.numel(), generator=g)[:per_class[c]]
            keep.append(pool[perm])
        out[d] = torch.cat(keep).sort().values
    return out, per_class


def build_domain_probe_split(source, seed=0, domains=PROBE_DOMAINS):
    """构造域探针的 train/val/test（各自内部均衡，互不重叠）。

    返回 {split: {'idx': {domain: LongTensor}, 'per_class': [...]}}，
    以及 {domain: (feature, gt_label, paths)}。
    """
    data = {d: load_domain(source, d) for d in domains}
    result = {}
    for split in ('train', 'val', 'test'):
        per_domain = {}
        for d in domains:
            feat, lab, paths = data[d]
            gidx = split_index(d, split, paths)
            per_domain[d] = (lab[gidx], gidx)
        idx, per_class = balanced_domain_indices(per_domain, split, seed=seed)
        result[split] = {'idx': idx, 'per_class': per_class}
    # 防泄漏断言：任意域内三个 split 的索引两两不相交
    for d in domains:
        s = [set(result[k]['idx'][d].tolist()) for k in ('train', 'val', 'test')]
        assert not (s[0] & s[1] or s[0] & s[2] or s[1] & s[2]), f'{d} 的划分有重叠'
    return result, data


def build_category_probe_split(source, domain):
    """类别探针：直接用已提交的确定性划分，返回 {split: LongTensor} 与数据。"""
    feat, lab, paths = load_domain(source, domain)
    idx = {s: split_index(domain, s, paths) for s in ('train', 'val', 'test')}
    sets = [set(v.tolist()) for v in idx.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), '划分有重叠'
    return idx, (feat, lab, paths)
