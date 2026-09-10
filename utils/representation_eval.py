"""表征评估的共享实现（只读）。

把类别条件域探针抽到这里，使 scripts/validate_stage1_followup.py 与
scripts/compare_runs.py 调用**同一份实现**，避免两个脚本的评估协议悄悄分叉。

⚠️ 纯评估工具：不训练、不修改模型、不调超参。真值仅用于离线诊断。
"""
import os
import sys

import torch

sys.path.append(os.getcwd())
from models.source_model import PACS_CLASSES
from utils.probes import fit_probe
from utils.probe_splits import PROBE_DOMAINS, load_domain, split_index
from scripts.validate_stage1 import extract


def class_conditional_domain(inn, dev, source, seed=0):
    """固定真实类别后再分域。类别在每个 probe 内恒定，**不存在类别/域混淆**。"""
    data = {d: load_domain(source, d) for d in PROBE_DOMAINS}
    reps = {}
    for d in PROBE_DOMAINS:
        f = data[d][0]
        zc, zs = extract(inn, f, dev)
        fc = f.detach().cpu()
        reps[d] = {'f': fc, 'z_c': zc, 'z_s': zs,
                   '||z_c||': zc.norm(dim=1, keepdim=True),
                   '||z_s||': zs.norm(dim=1, keepdim=True)}

    per_class = {}
    for ci, cname in enumerate(PACS_CLASSES):
        # 在每个 split 内部、对该类别独立做域均衡
        sel = {}
        for s in ('train', 'val', 'test'):
            pool = {}
            for d in PROBE_DOMAINS:
                g = split_index(d, s, data[d][2])
                pool[d] = g[data[d][1][g] == ci]
            m = min(len(v) for v in pool.values())
            picked = {}
            for d in PROBE_DOMAINS:
                gg = torch.Generator().manual_seed(seed * 100 + ci)
                perm = torch.randperm(len(pool[d]), generator=gg)[:m]
                picked[d] = pool[d][perm]
            sel[s] = (picked, m)
        entry = {'n_per_domain': {s: sel[s][1] for s in sel}}
        for key in ('f', 'z_c', 'z_s', '||z_c||', '||z_s||'):
            sp = {}
            for s in ('train', 'val', 'test'):
                xs, ys = [], []
                for di, d in enumerate(PROBE_DOMAINS):
                    ix = sel[s][0][d]
                    xs.append(reps[d][key][ix])
                    ys.append(torch.full((len(ix),), di, dtype=torch.long))
                sp[s] = (torch.cat(xs), torch.cat(ys))
            r, _ = fit_probe(key, sp, len(PROBE_DOMAINS), seed=seed)
            entry[key] = dict(train=r['train'], val=r['val'], test=r['test'],
                              best_weight_decay=r['best_weight_decay'])
        per_class[cname] = entry

    macro = {}
    for key in ('f', 'z_c', 'z_s', '||z_c||', '||z_s||'):
        t = torch.tensor([per_class[c][key]['test'] for c in PACS_CLASSES])
        macro[key] = dict(mean=t.mean().item(), std=t.std().item(),
                          values={c: per_class[c][key]['test'] for c in PACS_CLASSES})
    return dict(per_class=per_class, macro=macro, chance=1.0 / len(PROBE_DOMAINS),
                note=('每个 probe 内真实类别恒定，因此不存在任何类别/域混淆的可能。'
                      '域均衡在每个 split 内部、对每个类别独立进行。'))
