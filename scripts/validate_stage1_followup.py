"""Stage 1 验证的定向追加诊断（纯离线，不训练、不改动任何方法）。

回答三个问题：
  1. random swap 的 agreement=100% / KL≈0 是否由恒等或近恒等配对造成
  2. 类别探针 test 准确率的自助置信区间
  3. **类别条件下的域探针** —— 固定真实类别后，域信息还剩多少

⚠️ 不重跑完整验证、不覆盖原 validation JSON、不调探针超参、不改 Stage 1。
"""
import argparse
import json
import os
import sys

import torch

sys.path.append(os.getcwd())
from models.source_model import SourceModel, PACS_CLASSES
from utils.probes import fit_probe, Standardizer
from utils.probe_splits import PROBE_DOMAINS, build_category_probe_split, load_domain, split_index
from utils.target_cache import TargetFeatures
from scripts.validate_stage1 import load_frozen_inn, extract
from scripts.train_source import get_device


# ── 1-2. swap 非平凡性核实 + 高精度 KL/JS ────────────────────────────
@torch.no_grad()
def swap_verify(inn, C, dev, source, target, seed=0):
    ds = TargetFeatures(source, target).to(dev)
    zc, zs, _ = inn(ds.feature)
    lf = C(ds.feature)
    easy = ds.easy_mask.nonzero(as_tuple=True)[0]
    hard = ds.hard_mask.nonzero(as_tuple=True)[0]
    out = {}

    def run(tag, ia, ib):
        n = min(len(ia), len(ib))
        g = torch.Generator(device=dev).manual_seed(seed)      # 与主验证同种子同顺序
        a = ia[torch.randperm(len(ia), generator=g, device=dev)[:n]]
        b = ib[torch.randperm(len(ib), generator=g, device=dev)[:n]]
        f_swap = inn.inverse(zc[a], zs[b])
        lq = C(f_swap)
        lp_ = torch.log_softmax(lf[a], 1); lq_ = torch.log_softmax(lq, 1)
        p, q = lp_.exp(), lq_.exp()
        kl = (p * (lp_ - lq_)).sum(1).mean().item()
        m = (0.5 * (p + q)).clamp_min(1e-12).log()
        js = (0.5 * ((p * (lp_ - m)).sum(1) + (q * (lq_ - m)).sum(1))).mean().item()
        dz = (zs[a] - zs[b]).norm(dim=1)
        df = (ds.feature[a] - f_swap).norm(dim=1)
        out[tag] = dict(
            n=int(n), frac_a_eq_b=(a == b).float().mean().item(),
            agreement=(lf[a].argmax(1) == lq.argmax(1)).float().mean().item(),
            kl=kl, js=js,
            zs_diff_mean=dz.mean().item(), zs_diff_median=dz.median().item(),
            zs_diff_min=dz.min().item(),
            f_swap_diff_mean=df.mean().item(), f_swap_diff_median=df.median().item(),
            f_swap_diff_max=df.max().item(), f_swap_diff_min=df.min().item(),
            f_norm_mean=ds.feature[a].norm(dim=1).mean().item())

    allidx = torch.arange(len(ds), device=dev)
    run('random', allidx, allidx)
    run('easyA_hardB', easy, hard)
    run('hardA_easyB', hard, easy)
    return out


# ── 3. 类别探针的自助置信区间 ────────────────────────────────────────
def bootstrap_category(inn, dev, source, target, seed=0, n_boot=1000):
    """用与主验证**完全相同**的协议与种子重建同一个探针（确定性，非重新调参），
    仅为取得 test 上的逐样本预测以计算自助 CI。"""
    idx, (feat, gt, _) = build_category_probe_split(source, target)
    zc, zs = extract(inn, feat, dev)
    reps = {'f': feat.detach().cpu(), 'z_c': zc, 'z_s': zs}
    out = {}
    for name, x in reps.items():
        sp = {s: (x[idx[s]], gt[idx[s]]) for s in ('train', 'val', 'test')}
        r, frozen = fit_probe(name, sp, len(PACS_CLASSES), seed=seed)
        xt = frozen.st(sp['test'][0].float()); yt = sp['test'][1]
        with torch.no_grad():
            correct = frozen.lin(xt).argmax(1).eq(yt).float()
        g = torch.Generator().manual_seed(seed)
        n = correct.numel()
        boots = torch.stack([correct[torch.randint(0, n, (n,), generator=g)].mean()
                             for _ in range(n_boot)])
        lo, hi = torch.quantile(boots, torch.tensor([0.025, 0.975]))
        out[name] = dict(test=r['test'], ci95_low=lo.item(), ci95_high=hi.item(),
                         n_test=n, n_boot=n_boot, best_weight_decay=r['best_weight_decay'])
    return out


# ── 4-5. 类别条件下的域探针 ──────────────────────────────────────────
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-boot', type=int, default=1000)
    args = ap.parse_args()

    dev = get_device()
    inn, ck = load_frozen_inn(args.checkpoint, dev)
    cfg = ck['config']
    src, _ = SourceModel.load(f"checkpoints/source/{cfg['source']}.pth", device=dev)
    src.freeze(); C = src.classify
    source, target = cfg['source'], cfg['target']
    print(f"🔬 定向追加诊断 | {os.path.basename(args.checkpoint)}")
    print("   ⚠️ 不重跑完整验证、不覆盖原 JSON、不调超参、不改 Stage 1\n")

    rep = {'checkpoint': os.path.basename(args.checkpoint), 'seed': args.seed}
    print('1-2 swap 非平凡性核实 ...'); rep['swap_verify'] = swap_verify(inn, C, dev, source, target, args.seed)
    print('3   类别探针自助 CI ...');   rep['category_bootstrap'] = bootstrap_category(inn, dev, source, target, args.seed, args.n_boot)
    print('4-5 类别条件域探针 ...');    rep['class_conditional_domain'] = class_conditional_domain(inn, dev, source, args.seed)

    out = args.checkpoint.replace('.pth', '_validation_followup.json')
    with open(out, 'w') as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False)
    print(f"\n💾 {out}")


if __name__ == '__main__':
    main()
