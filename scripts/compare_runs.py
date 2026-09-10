"""跨 Stage 1 运行的统一评估（只读）。

⚠️ 本脚本**只读取 checkpoint 并评估**：不训练、不修改模型、不调超参、不实现
   任何损失。探针协议完全沿用 utils/probes.py 中已固定的两阶段流程
   （超参搜索只用 train/val，test 在 best_wd 固定后只评估一次）。

评估协议（固定，不随方法改动）
------------------------------
类别探针     cartoon 的 splits/ 确定性 80/10/10 划分
域探针       cartoon / photo / sketch，**类别条件**版本：固定真实类别后再分域，
             域均衡在每个 split 内部、对每个类别独立进行。每个 probe 内类别
             恒定，因此不存在类别/域混淆。

关于「留出」的准确表述
----------------------
类别条件域探针**必然读取 photo/sketch 的特征**（它要在三域间分类），所以只要
反复查看 Acc(d|z_c,y) 来筛选方法，photo/sketch 就已进入方法选择回路。

因此本套指标整体应视为 **exploration diagnostic**，而非未经触碰的 test。
真正保持留出的是另一件事：photo / sketch 作为**适应目标域**（跑整条
art_painting→photo / →sketch 管线）及其**下游任务准确率**——那些在方法定型
之前不应触碰。论文中须按此区分表述。

用法:
    python scripts/compare_runs.py checkpoints/stage1/*.pth
    python scripts/compare_runs.py A.pth B.pth --json out.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.append(os.getcwd())
from models.source_model import SourceModel, PACS_CLASSES
from utils.probes import fit_probe
from utils.probe_splits import PROBE_DOMAINS, build_category_probe_split
from utils.target_cache import TargetFeatures
from utils.stage1_losses import style_removal_ratio, tangent_orthogonality
from scripts.validate_stage1 import load_frozen_inn, extract
from utils.representation_eval import class_conditional_domain
from scripts.train_source import get_device

DIAG_SEED = 999999
CHANCE_CLASS = 1.0 / len(PACS_CLASSES)
CHANCE_DOMAIN = 1.0 / len(PROBE_DOMAINS)


def category_probes(inn, dev, source, target, seed):
    """Acc(y | f / z_c / z_s)。

    f 是**类别线性可分性的参考基线**，不是上界：z_c=g_c(f) 是非线性变换后的
    表征，完全可能在线性探针上超过 f（实测 89.36% > 88.51%）。
    """
    idx, (feat, gt, _) = build_category_probe_split(source, target)
    zc, zs = extract(inn, feat, dev)
    out = {}
    for name, x in (('f', feat.detach().cpu()), ('z_c', zc), ('z_s', zs)):
        sp = {s: (x[idx[s]], gt[idx[s]]) for s in ('train', 'val', 'test')}
        r, _ = fit_probe(name, sp, len(PACS_CLASSES), seed=seed)
        out[name] = dict(test=r['test'], val=r['val'], train=r['train'],
                         best_weight_decay=r['best_weight_decay'])
    return out


@torch.no_grad()
def _no_grad_geometry(inn, C, ds):
    zc, zs, _ = inn(ds.feature)
    fc = inn.inverse(zc, torch.zeros_like(zs))
    lf, lfc = C(ds.feature), C(fc)
    rt = inn.inverse(zc, zs)
    return dict(agreement=(lf.argmax(1) == lfc.argmax(1)).float().mean().item(),
                r_style=style_removal_ratio(ds.feature, fc).item(),
                zc_std=zc.std().item(), zs_std=zs.std().item(),
                roundtrip=(ds.feature - rt).abs().max().item())


def geometry(inn, C, dev, source, target):
    ds = TargetFeatures(source, target).to(dev)
    out = _no_grad_geometry(inn, C, ds)
    with torch.no_grad():
        zc, zs, _ = inn(ds.feature[:256])
    L, d = tangent_orthogonality(inn.inverse, zc, zs,
                                 generator=torch.Generator(device=dev).manual_seed(DIAG_SEED),
                                 detach_latent=True, return_diagnostics=True)
    c = d['abs_cos']
    out.update(L_orth=L.item(), cos_mean=c.mean().item(), cos_p90=torch.quantile(c, 0.9).item(),
               a_c=d['norm_a_c'].mean().item(), a_s=d['norm_a_s'].mean().item())
    return out


def evaluate_run(path, dev, seed):
    inn, ck = load_frozen_inn(path, dev)
    cfg = ck['config']
    src, _ = SourceModel.load(f"checkpoints/source/{cfg['source']}.pth", device=dev)
    src.freeze()
    cat = category_probes(inn, dev, cfg['source'], cfg['target'], seed)
    geo = geometry(inn, src.classify, dev, cfg['source'], cfg['target'])
    cc = class_conditional_domain(inn, dev, cfg['source'], seed=seed)
    return dict(
        name=os.path.basename(path).replace('.pth', ''),
        config=dict(lambda_flow=cfg['lambda_flow'], lambda_sem=cfg['lambda_sem'],
                    lambda_orth=cfg['lambda_orth'], n_c=cfg['n_c'], n_s=cfg['n_s'],
                    n_blocks=cfg['n_blocks'], steps=cfg['steps'], seed=cfg['seed'],
                    run_tag=ck.get('run_tag', ''),
                    batch_shuffle=ck.get('batch_shuffle_backend', 'unknown(pre-fix)')),
        category=cat, geometry=geo,
        domain_cond={k: cc['macro'][k] for k in ('f', 'z_c', 'z_s', '||z_c||', '||z_s||')},
        domain_cond_per_class_zc={c: cc['per_class'][c]['z_c']['test'] for c in PACS_CLASSES},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoints', nargs='+')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    dev = get_device()

    print('⚠️ 只读评估：不训练、不修改模型、不调超参。探针协议固定。')
    print('⚠️ 本套指标整体为 exploration diagnostic，非未经触碰的 test；'
          'photo/sketch 的特征已被类别条件域探针使用。')
    print(f'   保持留出的是 photo/sketch 作为**适应目标域**及其下游任务准确率。\n')

    rows = []
    for p in args.checkpoints:
        if not os.path.exists(p):
            print(f'  ⚠️ 跳过不存在的 {p}'); continue
        print(f'  评估 {os.path.basename(p)} ...', flush=True)
        rows.append(evaluate_run(p, dev, args.seed))

    # 去掉所有 run 名的公共前缀，避免表格里被截断成无法区分
    if len(rows) > 1:
        names = [r['name'] for r in rows]
        pre = os.path.commonprefix(names)
        pre = pre[:pre.rfind('_') + 1] if '_' in pre else ''
        for r in rows:
            r['short'] = r['name'][len(pre):] or r['name']
    else:
        rows[0]['short'] = rows[0]['name']

    hdr = (f"\n{'run':<34}{'λo':>5}{'Acc(y|f)':>10}{'Acc(y|z_c)':>12}{'Acc(y|z_s)':>12}"
           f"{'Acc(d|z_c,y)':>14}{'Acc(d|z_s,y)':>14}")
    print('=' * len(hdr.strip()) )
    print('类别探针 = cartoon test；域探针 = 类别条件 macro')
    print(f'随机基线：类别 {CHANCE_CLASS*100:.2f}%   域 {CHANCE_DOMAIN*100:.2f}%')
    print(hdr)
    print('-' * (len(hdr.strip()) + 2))
    for r in rows:
        print(f"  {r['short'][:32]:<32}{r['config']['lambda_orth']:>5.0f}"
              f"{r['category']['f']['test']*100:>9.2f}%{r['category']['z_c']['test']*100:>11.2f}%"
              f"{r['category']['z_s']['test']*100:>11.2f}%"
              f"{r['domain_cond']['z_c']['mean']*100:>13.2f}%"
              f"{r['domain_cond']['z_s']['mean']*100:>13.2f}%")

    print(f"\n{'run':<34}{'||z_c||→d':>11}{'||z_s||→d':>11}{'L_orth':>12}{'|cos|':>10}"
          f"{'一致率':>9}{'r_style':>10}{'往返':>11}")
    print('-' * 108)
    for r in rows:
        g = r['geometry']
        print(f"  {r['short'][:32]:<32}{r['domain_cond']['||z_c||']['mean']*100:>10.2f}%"
              f"{r['domain_cond']['||z_s||']['mean']*100:>10.2f}%{g['L_orth']:>12.3e}"
              f"{g['cos_mean']:>10.5f}{g['agreement']*100:>8.2f}%{g['r_style']:>10.5f}"
              f"{g['roundtrip']:>11.2e}")

    print(f"\n逐类别 Acc(d | z_c, y)")
    print(f"  {'run':<32}" + ''.join(f'{c[:8]:>10}' for c in PACS_CLASSES))
    for r in rows:
        print(f"  {r['short'][:30]:<32}"
              + ''.join(f"{r['domain_cond_per_class_zc'][c]*100:>9.2f}%" for c in PACS_CLASSES))

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump({'protocol': {
                'category_probe': 'cartoon splits/ 80-10-10',
                'domain_probe': 'class-conditional, balanced within each split',
                'probe_protocol': 'utils.probes.fit_probe (two-phase, test evaluated once)',
                'status': 'exploration diagnostic — NOT an untouched test set',
                'held_out': 'photo/sketch as adaptation targets and their end-task accuracy',
                'chance_class': CHANCE_CLASS, 'chance_domain': CHANCE_DOMAIN},
                'runs': rows}, fh, indent=2, ensure_ascii=False)
        print(f'\n💾 {args.json}')


if __name__ == '__main__':
    main()
