"""Stage 1 表征验证（纯离线评估，不训练、不改动 Stage 1）。

回答 research_idea.pdf 的核心问题：可逆解耦是否真的产生了
「底流形语义 z_c / 纤维风格 z_s」的结构。

⚠️ 本脚本**只报告结果，不把期望结论编码进判定逻辑**。若期望的大小关系不成立，
   如实报告，不修改损失、不重训 Stage 1、不新增机制。

真值使用边界：真值仅作为探针的标签与 C(f) 的准确率参考，全部经
load_target_cache(include_eval_only=True) 显式取出；不参与表征学习、不参与
探针超参选择之外的任何环节。

用法:
    python scripts/validate_stage1.py --checkpoint checkpoints/stage1/....pth
"""
import argparse
import json
import os
import sys

import torch

sys.path.append(os.getcwd())
from models.source_model import SourceModel, PACS_CLASSES
from models.feature_inn import FeatureINN
from utils.probes import fit_probe, probe_representation
from utils.probe_splits import (PROBE_DOMAINS, build_category_probe_split,
                                build_domain_probe_split, load_domain, split_index,
                                balanced_domain_indices)
from utils.stage1_losses import style_removal_ratio, tangent_orthogonality
from utils.target_cache import TargetFeatures, median_entropy_split
from scripts.train_source import get_device

ALL_DOMAINS = ('art_painting', 'cartoon', 'photo', 'sketch')
K_DIR = 64            # 随机切子空间的方向数
DIAG_SEED = 999999


# ───────────────────────── §0-1 加载与表征提取 ─────────────────────────

def load_frozen_inn(path, dev):
    ck = torch.load(path, map_location=dev)
    cfg = ck['config']
    inn = FeatureINN(dim=cfg['feature_dim'], n_c=cfg['n_c'], n_blocks=cfg['n_blocks']).to(dev)
    inn.load_state_dict(ck['inn'])
    inn.eval()
    for p in inn.parameters():
        p.requires_grad_(False)            # 普通提取路径不保留任何参数梯度
    return inn, ck


@torch.no_grad()
def extract(inn, feature, dev, chunk=2048):
    """提取 latent 并立即搬到 CPU；探针永远拿不到通向 INN 的 autograd 路径。"""
    zc, zs = [], []
    for i in range(0, feature.shape[0], chunk):
        a, b, _ = inn(feature[i:i + chunk].to(dev))
        zc.append(a.detach().cpu()); zs.append(b.detach().cpu())
    return torch.cat(zc), torch.cat(zs)


# ───────────────────────── §2 重构诊断 ─────────────────────────

@torch.no_grad()
def reconstruction_diag(inn, dev, source):
    qs = torch.tensor([0.5, 0.95, 0.99])
    out = {}
    for d in ALL_DOMAINS:
        ds = TargetFeatures(source, d).to(dev)
        zc, zs, ld = inn(ds.feature)
        rt = inn.inverse(zc, zs)
        per_abs = (ds.feature - rt).abs().amax(1)
        per_rel = (ds.feature - rt).norm(dim=1) / ds.feature.norm(dim=1)
        qa = torch.quantile(per_abs, qs.to(dev)); qr = torch.quantile(per_rel, qs.to(dev))
        out[d] = dict(n=len(ds),
                      abs_median=qa[0].item(), abs_p95=qa[1].item(),
                      abs_p99=qa[2].item(), abs_max=per_abs.max().item(),
                      rel_median=qr[0].item(), rel_p95=qr[1].item(),
                      rel_p99=qr[2].item(), rel_max=per_rel.max().item(),
                      logdet_finite=bool(torch.isfinite(ld).all()),
                      latent_finite=bool(torch.isfinite(zc).all() and torch.isfinite(zs).all()),
                      zc_mean=zc.mean().item(), zc_std=zc.std().item(),
                      zs_mean=zs.mean().item(), zs_std=zs.std().item())
    return out


# ───────────────────────── §3 类别探针 ─────────────────────────

def category_probes(inn, C, dev, source, target, seed=0):
    idx, (feat, gt, _paths) = build_category_probe_split(source, target)
    zc, zs = extract(inn, feat, dev)
    f_cpu = feat.detach().cpu()

    # C(f)：冻结源分类器在 test 上的真值准确率——这是 Stage 1 要保持的
    # **原始 source decision 参考**，与 LinearProbe(f) 不是一回事。
    with torch.no_grad():
        pred = C(feat.to(dev)).argmax(1).cpu()
    src_acc = {s: pred[idx[s]].eq(gt[idx[s]]).float().mean().item() for s in idx}

    reps = {'f': f_cpu, 'z_c': zc, 'z_s': zs}
    probes = {}
    for name, x in reps.items():
        sp = {s: (x[idx[s]], gt[idx[s]]) for s in ('train', 'val', 'test')}
        probes[name] = probe_representation(name, sp, len(PACS_CLASSES), seed=seed)
    return dict(source_classifier_acc=src_acc, linear_probes=probes,
                n={s: int(len(idx[s])) for s in idx},
                note=('C(f) 是原始 source decision 参考；LinearProbe(f) 只是 '
                      '2048 维特征线性可分性的参考，两者不可混用。'))


# ───────────────────────── §4 域探针 ─────────────────────────

def _domain_tensors(idx_map, data, reps_by_domain, key):
    xs, ys = [], []
    for di, d in enumerate(PROBE_DOMAINS):
        sel = idx_map[d]
        xs.append(reps_by_domain[d][key][sel])
        ys.append(torch.full((len(sel),), di, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


def domain_probes(inn, dev, source, seed=0, n_resample=10):
    res, data = build_domain_probe_split(source, seed=seed)
    reps = {}
    for d in PROBE_DOMAINS:
        feat = data[d][0]
        zc, zs = extract(inn, feat, dev)
        f_cpu = feat.detach().cpu()
        reps[d] = {'f': f_cpu, 'z_c': zc, 'z_s': zs,
                   '||f||': f_cpu.norm(dim=1, keepdim=True),        # 严格 [N,1]
                   '||z_c||': zc.norm(dim=1, keepdim=True),
                   '||z_s||': zs.norm(dim=1, keepdim=True)}

    out = {'per_class_counts': {s: res[s]['per_class'] for s in res},
           'n': {s: int(sum(len(v) for v in res[s]['idx'].values())) for s in res},
           'chance': 1.0 / len(PROBE_DOMAINS), 'primary': {}, 'robustness': {}}

    for key in ('f', 'z_c', 'z_s', '||f||', '||z_c||', '||z_s||'):
        sp = {s: _domain_tensors(res[s]['idx'], data, reps, key)
              for s in ('train', 'val', 'test')}
        r, frozen = fit_probe(key, sp, len(PROBE_DOMAINS), seed=seed)
        out['primary'][key] = r
        # robustness：只重采样 balanced test 子集，复用同一个冻结探针
        accs = []
        for rs in range(1, n_resample + 1):     # primary 用 seed 0，robustness 用 1..n，互不重叠
            per_domain = {}
            for d in PROBE_DOMAINS:
                g = split_index(d, 'test', data[d][2])
                per_domain[d] = (data[d][1][g], g)
            ridx, _ = balanced_domain_indices(per_domain, 'test', seed=rs)
            x, y = _domain_tensors(ridx, data, reps, key)
            accs.append(frozen.accuracy(x, y))
        t = torch.tensor(accs)
        out['robustness'][key] = dict(mean=t.mean().item(), std=t.std().item(),
                                      values=accs, n_resample=n_resample,
                                      seeds=list(range(1, n_resample + 1)),
                                      note=('仅重采样 balanced test 子集（seed 1..N），'
                                            '探针、标准化与超参全部保持冻结；primary 用 '
                                            'seed 0，与这些重采样完全独立。度量的是 test '
                                            '采样方差，不含训练随机性或超参选择。'))

    # 次要参考：不均衡版本（类别先验混淆）
    unb = {}
    for key in ('f', 'z_c', 'z_s'):
        sp = {}
        for s in ('train', 'val', 'test'):
            xs, ys = [], []
            for di, d in enumerate(PROBE_DOMAINS):
                g = split_index(d, s, data[d][2])
                xs.append(reps[d][key][g]); ys.append(torch.full((len(g),), di, dtype=torch.long))
            sp[s] = (torch.cat(xs), torch.cat(ys))
        unb[key] = probe_representation(key, sp, len(PROBE_DOMAINS), seed=seed)
    # 不均衡版本的域样本数与多数域基线（raw accuracy 不可与 33.33% 比较）
    unb_counts, maj = {}, {}
    for s_ in ('train', 'val', 'test'):
        cnt = [len(split_index(d, s_, data[d][2])) for d in PROBE_DOMAINS]
        unb_counts[s_] = dict(zip(PROBE_DOMAINS, cnt))
        maj[s_] = max(cnt) / sum(cnt)
    out['unbalanced_reference'] = dict(
        probes=unb,
        domain_counts=unb_counts, majority_domain_baseline=maj,
        label='class-prior-confounded reference',
        note=('各域类别分布差异极大（sketch 的 house 占 2.0%，photo 占 16.8%），'
              '仅含类别信息的表征也能靠类别先验预测域。**不得**用于支持主要的'
              '域解耦主张。另外三域样本数不等，raw accuracy 的参考基线是上面的 '
              'majority_domain_baseline，而非 33.33%。'))
    return out


# ───────────────────────── §5-6 swap / 风格移除 ─────────────────────────

def _dists(p_logits, q_logits):
    lp = torch.log_softmax(p_logits, 1); lq = torch.log_softmax(q_logits, 1)
    p, q = lp.exp(), lq.exp()
    kl = (p * (lp - lq)).sum(1).mean().item()
    m = (0.5 * (p + q)).clamp_min(1e-12).log()
    js = (0.5 * ((p * (lp - m)).sum(1) + (q * (lq - m)).sum(1))).mean().item()
    agree = (p_logits.argmax(1) == q_logits.argmax(1)).float().mean().item()
    return dict(agreement=agree, kl=kl, js=js)


@torch.no_grad()
def swap_diag(inn, C, dev, source, target, seed=0):
    ds = TargetFeatures(source, target).to(dev)
    zc, zs, _ = inn(ds.feature)
    lf = C(ds.feature)
    easy = ds.easy_mask.nonzero(as_tuple=True)[0]
    hard = ds.hard_mask.nonzero(as_tuple=True)[0]
    out = {}

    def run(tag, ia, ib):
        n = min(len(ia), len(ib))
        g = torch.Generator(device=dev).manual_seed(seed)
        a = ia[torch.randperm(len(ia), generator=g, device=dev)[:n]]
        b = ib[torch.randperm(len(ib), generator=g, device=dev)[:n]]
        f_swap = inn.inverse(zc[a], zs[b])          # A 的语义 + B 的风格
        r = _dists(lf[a], C(f_swap)); r['n'] = int(n); out[tag] = r

    allidx = torch.arange(len(ds), device=dev)
    run('random', allidx, allidx)
    run('easyA_hardB', easy, hard)
    run('hardA_easyB', hard, easy)
    out['note'] = 'swap 的构造与指标均不使用真值'
    return out


@torch.no_grad()
def style_removal_diag(inn, C, dev, source, target):
    ds = TargetFeatures(source, target).to(dev)
    zc, zs, _ = inn(ds.feature)
    fc = inn.inverse(zc, torch.zeros_like(zs))
    lf, lfc = C(ds.feature), C(fc)
    out = {}
    for tag, m in (('all', torch.ones(len(ds), dtype=torch.bool, device=dev)),
                   ('easy', ds.easy_mask), ('hard', ds.hard_mask)):
        r = _dists(lf[m], lfc[m])
        r.update(n=int(m.sum()),
                 r_style=style_removal_ratio(ds.feature[m], fc[m]).item(),
                 mean_df=(ds.feature[m] - fc[m]).norm(dim=1).mean().item())
        out[tag] = r
    out['tau'] = ds.tau.item()
    return out


# ───────────────────────── §7 正交性诊断 ─────────────────────────

def orthogonality_diag(inn, dev, source, target, n_samples=8, k=K_DIR, seed=DIAG_SEED):
    """随机 |cos| 统计 + **随机切子空间角度诊断**。

    ⚠️ 后者只在各随机采样的 k 维子空间上计算主角度，**不是**完整
       1536 维语义切空间与 512 维风格切空间的精确主角度谱。它近似刻画
       局部几何分离程度，措辞上不得称为 exact principal angles。
    """
    ds = TargetFeatures(source, target).to(dev)
    with torch.no_grad():
        zc_all, zs_all, _ = inn(ds.feature[:256])
    # 诊断路径只取前向值、完全不反传，故用 detach_latent=True 让守卫放行。
    # （冻结 INN 后 z_c/z_s 本就不带计算图；实测完整图版与截断版的 forward
    #   值逐位相同 |Δ|=0，两者仅在参数梯度方向上不同，与本诊断无关。）
    L, d = tangent_orthogonality(inn.inverse, zc_all, zs_all,
                                 generator=torch.Generator(device=dev).manual_seed(seed),
                                 detach_latent=True, return_diagnostics=True)
    c = d['abs_cos']
    out = dict(L_orth=L.item(), cos_mean=c.mean().item(), cos_median=c.median().item(),
               cos_p90=torch.quantile(c, 0.9).item(), cos_max=c.max().item(),
               a_c=d['norm_a_c'].mean().item(), a_s=d['norm_a_s'].mean().item())

    svs, angs = [], []
    with torch.no_grad():
        base_zc, base_zs, _ = inn(ds.feature[:n_samples])
    for i in range(n_samples):
        g = torch.Generator(device=dev).manual_seed(seed + i)
        zc_r = base_zc[i].unsqueeze(0).expand(k, -1).contiguous()
        zs_r = base_zs[i].unsqueeze(0).expand(k, -1).contiguous()
        W = torch.randn(k, zc_r.shape[1], generator=g, device=dev)
        V = torch.randn(k, zs_r.shape[1], generator=g, device=dev)
        W /= W.norm(dim=1, keepdim=True); V /= V.norm(dim=1, keepdim=True)
        A_c = torch.func.jvp(lambda t: inn.inverse(t, zs_r), (zc_r,), (W,))[1].T  # [2048,k]
        A_s = torch.func.jvp(lambda t: inn.inverse(zc_r, t), (zs_r,), (V,))[1].T
        Qc, _ = torch.linalg.qr(A_c.cpu()); Qs, _ = torch.linalg.qr(A_s.cpu())
        s = torch.linalg.svdvals(Qc.T @ Qs).clamp(0, 1)
        svs.append(s); angs.append(torch.arccos(s))
    S = torch.stack(svs); A = torch.stack(angs)
    out['randomized_subspace_angles'] = dict(
        k_directions=k, n_samples=n_samples,
        singular_max=S.max().item(), singular_median=S.median().item(),
        angle_min_deg=torch.rad2deg(A.min()).item(),
        angle_median_deg=torch.rad2deg(A.median()).item(),
        angle_max_deg=torch.rad2deg(A.max()).item(),
        note=('randomized tangent-subspace angle diagnostic：只在各采样的 '
              f'{k} 维子空间上计算，近似刻画局部几何分离；**不是**完整 '
              '1536D/512D 切空间的精确主角度谱。'))
    return out


# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-resample', type=int, default=10)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    dev = get_device()
    inn, ck = load_frozen_inn(args.checkpoint, dev)
    cfg = ck['config']
    src_model, src_meta = SourceModel.load(f"checkpoints/source/{cfg['source']}.pth", device=dev)
    src_model.freeze()
    C = src_model.classify
    source, target = cfg['source'], cfg['target']

    print(f"🔍 Stage 1 表征验证 | checkpoint {os.path.basename(args.checkpoint)}")
    print(f"   {source} → {target} | n_c={cfg['n_c']} n_s={cfg['n_s']} "
          f"blocks={cfg['n_blocks']} λ_orth={cfg['lambda_orth']} seed={cfg['seed']}")
    print("   ⚠️ 纯离线评估：不训练、不改动 Stage 1；结果原样报告\n")

    report = {
        'checkpoint': os.path.basename(args.checkpoint),
        'stage1_config': cfg, 'source_meta': src_meta, 'validation_seed': args.seed,
        'limitation': ('Stage 1 的 INN 在**全部** cartoon 特征（含 probe 的 test 划分）'
                       '上训练过，但从未使用目标域真值。这属于 target-unlabeled / '
                       'transductive 设定，与当前 SFDA 设定一致。更严格的 inductive '
                       '版本（只用 cartoon_train 训练 INN）留作后续 robustness 实验。'),
    }
    print('§2 重构诊断 ...');        report['reconstruction'] = reconstruction_diag(inn, dev, source)
    print('§3 类别探针 ...');        report['category'] = category_probes(inn, C, dev, source, target, args.seed)
    print('§4 域探针 ...');          report['domain'] = domain_probes(inn, dev, source, args.seed, args.n_resample)
    print('§5 swap 一致性 ...');     report['swap'] = swap_diag(inn, C, dev, source, target, args.seed)
    print('§6 风格移除 ...');        report['style_removal'] = style_removal_diag(inn, C, dev, source, target)
    print('§7 正交性 ...');          report['orthogonality'] = orthogonality_diag(inn, dev, source, target)

    out = args.out or args.checkpoint.replace('.pth', '_validation.json')
    with open(out, 'w') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\n💾 {out}")
    return report


if __name__ == '__main__':
    main()
