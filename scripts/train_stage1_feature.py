"""Stage 1：特征空间 INN 训练（research_idea.pdf 的可逆解耦部分）。

    L_total = λ_flow·L_flow + λ_sem·L_sem + λ_orth·L_orth

首版正式权重 λ_flow=1, λ_sem=1, λ_orth=25，由 cartoon 上的受控消融确定
（3 seed × {0,25,50} × 1000 步）：λ=25 保留了 λ=50 大部分几何抑制
（84.1% vs 90.0%），同时 orth/base 一致更低（0.92±0.56 vs 2.65±0.37）、
L_flow 代价更小（1.93% vs 2.50%）。不再进一步细调，避免对单一目标域过拟合。

约束：
  * 源模型 F 与分类器 C 全程冻结
  * 不使用目标域真值，不做伪标签监督
  * Stage 1 的损失不区分 easy/hard；难易身份保留给后续的风格融合阶段
  * 固定步数训练，不用目标域标签做 checkpoint 选择

用法:
    python scripts/train_stage1_feature.py --target cartoon
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.append(os.getcwd())
from models.source_model import SourceModel, PACS_DOMAINS
from models.feature_inn import FeatureINN
from utils.target_cache import TargetFeatures
from utils.stage1_losses import (flow_nll, flow_nll_terms, semantic_consistency,
                                 style_removal_ratio, tangent_orthogonality)
from scripts.train_source import get_device

CKPT_DIR = 'checkpoints/stage1'
DIAG_DIR_SEED = 999999      # 诊断用固定切方向种子，使 L_orth 轨迹跨步骤可比


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ckpt_name(a):
    """checkpoint 命名：只写入已计划做消融的项（n_c、λ_orth、seed）。

    n_blocks/steps 等当前固定的配置不进文件名，完整配置存在 checkpoint 内。
    """
    return f'{a.source}__{a.target}__nc{a.n_c}_lo{a.lambda_orth:g}_s{a.seed}.pth'


def evaluate(inn, C, ds, dev, seed=DIAG_DIR_SEED, n_orth=256):
    """全量诊断。L_orth 需要梯度图，故不能整体包在 no_grad 内。"""
    inn.eval()
    with torch.no_grad():
        zc, zs, ld = inn(ds.feature)
        fc = inn.inverse(zc, torch.zeros_like(zs))
        rt = inn.inverse(zc, zs)
        lf, lfc = C(ds.feature), C(fc)
        quad, logdet = flow_nll_terms(zc, zs, ld)
        out = dict(L_flow=flow_nll(zc, zs, ld).item(),
                   L_sem=semantic_consistency(lf, lfc).item(),
                   agree=(lf.argmax(1) == lfc.argmax(1)).float().mean().item(),
                   r_style=style_removal_ratio(ds.feature, fc).item(),
                   quad=quad.item(), logdet=logdet.item(),
                   zc_std=zc.std().item(), zs_std=zs.std().item(),
                   zc_mean=zc.mean().item(), zs_mean=zs.mean().item(),
                   zs_norm=zs.norm(dim=1).mean().item(),
                   df=(ds.feature - fc).norm(dim=1).mean().item(),
                   roundtrip=(ds.feature - rt).abs().max().item())
    zc, zs, _ = inn(ds.feature[:n_orth])
    L, d = tangent_orthogonality(inn.inverse, zc, zs,
                                 generator=torch.Generator(device=dev).manual_seed(seed),
                                 return_diagnostics=True)
    inn.zero_grad(set_to_none=True); inn.train()
    c = d['abs_cos']
    out.update(L_orth=L.item(), cos_mean=c.mean().item(),
               cos_p90=torch.quantile(c, 0.9).item(),
               a_c=d['norm_a_c'].mean().item(), a_s=d['norm_a_s'].mean().item())
    return out


def fmt(tag, r):
    return (f"{tag:>7} L_flow={r['L_flow']:>9.5f} L_sem={r['L_sem']:>9.3e} "
            f"L_orth={r['L_orth']:>9.3e} 一致率={r['agree']*100:>6.2f}% "
            f"r_style={r['r_style']:.5f} |cos|={r['cos_mean']:.5f}/{r['cos_p90']:.5f} "
            f"‖a‖={r['a_c']:.4f}/{r['a_s']:.4f} z_std={r['zc_std']:.4f}/{r['zs_std']:.4f} "
            f"rt={r['roundtrip']:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='art_painting', choices=PACS_DOMAINS)
    ap.add_argument('--target', default='cartoon', choices=PACS_DOMAINS)
    ap.add_argument('--n-c', type=int, default=1536)
    ap.add_argument('--n-blocks', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--lambda-flow', type=float, default=1.0)
    ap.add_argument('--lambda-sem', type=float, default=1.0)
    ap.add_argument('--lambda-orth', type=float, default=25.0)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--no-save', action='store_true')
    args = ap.parse_args()

    set_all_seeds(args.seed)
    dev = get_device()
    src, src_meta = SourceModel.load(f'checkpoints/source/{args.source}.pth', device=dev)
    src.freeze()
    C = src.classify
    ds = TargetFeatures(args.source, args.target).to(dev)
    inn = FeatureINN(dim=ds.dim, n_c=args.n_c, n_blocks=args.n_blocks).to(dev)
    n_s = ds.dim - args.n_c

    print(f'🖥️  {dev} | {ds.summary()}')
    print(f'📐 INN dim={ds.dim} n_c={args.n_c} n_s={n_s} blocks={args.n_blocks} '
          f'| 参数 {sum(p.numel() for p in inn.parameters()):,}')
    print(f'⚖️  λ_flow={args.lambda_flow} λ_sem={args.lambda_sem} λ_orth={args.lambda_orth} '
          f'| steps={args.steps} bs={args.batch_size} lr={args.lr} seed={args.seed}')
    print('🔒 F 与 C 冻结 | 不使用目标域真值或伪标签监督 | 损失不区分 easy/hard')
    print('   固定步数训练，不用目标域标签做 checkpoint 选择\n')

    # ActNorm 用真实目标 batch 初始化（必须在任何 optimizer 更新之前）
    inn.train()
    with torch.no_grad():
        inn(ds.feature[:args.batch_size])

    fp0 = torch.cat([p.flatten() for p in src.parameters()]).clone()
    with torch.no_grad():
        teacher0 = C(ds.feature).clone()
    before = evaluate(inn, C, ds, dev)
    print(fmt('before', before) + '\n')

    opt = torch.optim.Adam(inn.parameters(), lr=args.lr)
    data_gen = torch.Generator(device=dev); data_gen.manual_seed(args.seed)  # 确定性 batch 顺序
    dir_gen = torch.Generator(device=dev)                                    # 逐步确定性切方向

    step, t0 = 0, time.time()
    while step < args.steps:
        for fb, _hb, _idx in ds.iter_batches(args.batch_size, generator=data_gen,
                                             drop_last=True):
            dir_gen.manual_seed(args.seed * 100000 + step)
            zc, zs, ld = inn(fb)
            fc = inn.inverse(zc, torch.zeros_like(zs))
            Lf = flow_nll(zc, zs, ld)
            Ls = semantic_consistency(C(fb), C(fc))
            Lo = tangent_orthogonality(inn.inverse, zc, zs, generator=dir_gen)
            loss = args.lambda_flow * Lf + args.lambda_sem * Ls + args.lambda_orth * Lo

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(inn.parameters(), 1e9)   # 只测量，不裁剪
            opt.step()
            step += 1

            if step % args.log_every == 0 or step == args.steps:
                r = evaluate(inn, C, ds, dev)
                print(fmt(str(step), r) + f" gnorm={gn.item():.4f}")
            if step >= args.steps:
                break

    dt = time.time() - t0
    after = evaluate(inn, C, ds, dev)
    print(f"\n{'='*100}")
    print(f"用时 {dt:.0f}s ({dt/args.steps*1000:.0f} ms/步)")
    print(f"  往返 max|Δ| = {after['roundtrip']:.3e} {'✅' if after['roundtrip'] < 1e-5 else '❌'}")
    cls_same = torch.equal(fp0, torch.cat([p.flatten() for p in src.parameters()]))
    with torch.no_grad():
        teacher1 = C(ds.feature)
    teacher_delta = (teacher0 - teacher1).abs().max().item()
    print(f"  分类器参数逐位未变: {cls_same} {'✅' if cls_same else '❌'}")
    print(f"  teacher C(f) 未变 max|Δ| = {teacher_delta:.3e} "
          f"{'✅' if teacher_delta == 0.0 else '❌'}")
    print(f"  INN 参数全部有限: {all(torch.isfinite(p).all() for p in inn.parameters())} ✅")
    print(f"\n  {'量':<20}{'before':>14}{'after':>14}")
    for k, n in [('L_flow','L_flow'), ('L_sem','L_sem'), ('L_orth','L_orth'),
                 ('agree','语义一致率'), ('r_style','r_style'), ('cos_mean','|cos| mean'),
                 ('cos_p90','|cos| p90'), ('a_c','‖a_c‖'), ('a_s','‖a_s‖'),
                 ('zc_std','z_c std'), ('zs_std','z_s std')]:
        f = '{:>14.3e}' if abs(after[k]) < 1e-2 and after[k] != 0 else '{:>14.5f}'
        print(f"  {n:<20}" + f.format(before[k]) + f.format(after[k]))

    if not args.no_save:
        os.makedirs(CKPT_DIR, exist_ok=True)
        path = os.path.join(CKPT_DIR, ckpt_name(args))
        torch.save({
            'inn': inn.state_dict(),
            'config': {**vars(args), 'n_s': n_s, 'feature_dim': ds.dim,
                       'device': str(dev), 'steps_completed': step},
            'source_domain': args.source, 'target_domain': args.target,
            'seed': args.seed,
            'lambdas': {'flow': args.lambda_flow, 'sem': args.lambda_sem,
                        'orth': args.lambda_orth},
            'split': {'n_c': args.n_c, 'n_s': n_s},
            'entropy_split': {'tau': ds.tau.item(),
                              'n_easy': int(ds.easy_mask.sum()),
                              'n_hard': int(ds.hard_mask.sum()),
                              'rule': 'median predictive entropy (target-only)'},
            'source_meta': src_meta,
            'target_cache_meta': ds.meta,
            'metrics': {'before': before, 'after': after},
            'integrity': {'classifier_unchanged': cls_same,
                          'teacher_max_delta': teacher_delta,
                          'roundtrip_max_err': after['roundtrip']},
        }, path)
        print(f"\n💾 {path}")
        with open(path.replace('.pth', '.json'), 'w') as fh:
            json.dump({'config': {**vars(args), 'n_s': n_s},
                       'integrity': {'classifier_unchanged': cls_same,
                                     'teacher_max_delta': teacher_delta,
                                     'roundtrip_max_err': after['roundtrip']},
                       'metrics': {'before': before, 'after': after}},
                      fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
