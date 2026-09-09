"""难度代理诊断（纯离线评估，不改动任何适应方法）。

回答一个问题：源模型的不确定性能否作为 research_idea.pdf 中
「困难数据 X_h / 简单数据 X_e」划分的可靠依据？

⚠️ 本脚本使用目标域真值，**仅用于离线评估**。它不产生任何训练划分、
   不写回缓存、不参与模型选择。适应流程仍然只能通过
   load_target_cache()（默认剥离 eval_only）访问数据。

用法:
    python scripts/diagnose_difficulty.py --source art_painting --target cartoon
"""
import argparse
import os
import sys

import torch

sys.path.append(os.getcwd())
from models.source_model import PACS_CLASSES, PACS_DOMAINS
from scripts.cache_features import load_target_cache


def auroc(score, positive):
    """P(score[正类] > score[负类])，秩和法，无需 sklearn。positive=True 表示「预测错误」。"""
    pos, neg = score[positive], score[~positive]
    if pos.numel() == 0 or neg.numel() == 0:
        return float('nan')
    ranks = torch.empty_like(score)
    ranks[score.argsort()] = torch.arange(1, score.numel() + 1, dtype=score.dtype)
    return ((ranks[positive].sum() - pos.numel() * (pos.numel() + 1) / 2)
            / (pos.numel() * neg.numel())).item()


def binned(score, correct, edges, label):
    """按 score 的分位数分箱，报告每箱准确率。"""
    print(f'\n  {label} 分位区间 → 准确率')
    print(f'    {"区间":<12}{"n":>6}{"准确率":>10}{"score 范围":>22}')
    qs = torch.quantile(score, torch.tensor(edges))
    for i in range(len(edges) - 1):
        lo, hi = qs[i], qs[i + 1]
        m = (score >= lo) & (score <= hi) if i == len(edges) - 2 else (score >= lo) & (score < hi)
        if m.sum() == 0:
            continue
        print(f'    {int(edges[i]*100):>3}-{int(edges[i+1]*100):<3}%     '
              f'{int(m.sum()):>6}{correct[m].float().mean()*100:>9.2f}%'
              f'{f"[{lo:.3f}, {hi:.3f}]":>22}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='art_painting', choices=PACS_DOMAINS)
    ap.add_argument('--target', required=True, choices=PACS_DOMAINS)
    args = ap.parse_args()

    path = f'cache/features/{args.source}__{args.target}.pt'
    d = load_target_cache(path, include_eval_only=True)   # 显式取真值：仅评估
    prob, pseudo = d['prob'], d['pseudo_label']
    gt = d['eval_only']['label']
    correct = (pseudo == gt)
    n = correct.numel()

    # 三个候选难度分数：值越大 = 越「难」
    top2 = prob.topk(2, dim=1).values
    scores = {
        'predictive entropy': -(prob * prob.clamp_min(1e-12).log()).sum(1),
        '1 - max softmax': 1 - prob.max(1).values,
        'top1-top2 margin(取负)': -(top2[:, 0] - top2[:, 1]),
    }

    print('=' * 78)
    print(f'难度代理诊断  {args.source} → {args.target}')
    print(f'  N={n}  整体准确率={correct.float().mean()*100:.2f}%  '
          f'错误={int((~correct).sum())} 正确={int(correct.sum())}')
    print('  ⚠️ 真值仅用于本诊断，未参与任何训练或划分')
    print('=' * 78)

    ent = scores['predictive entropy']
    conf = prob.max(1).values

    binned(ent, correct, [0, .10, .25, .50, .75, .90, 1.0], '熵')
    binned(-conf, correct, [0, .10, .25, .50, .75, .90, 1.0], '最大 softmax 置信度(由高到低)')

    q25, q75 = torch.quantile(ent, torch.tensor([0.25, 0.75]))
    lo_m, hi_m = ent <= q25, ent >= q75
    print(f'\n  最低熵 25% 的准确率 : {correct[lo_m].float().mean()*100:>6.2f}%  (n={int(lo_m.sum())})')
    print(f'  最高熵 25% 的准确率 : {correct[hi_m].float().mean()*100:>6.2f}%  (n={int(hi_m.sum())})')
    print(f'  正确预测的平均熵    : {ent[correct].mean():>6.3f}')
    print(f'  错误预测的平均熵    : {ent[~correct].mean():>6.3f}')

    print(f'\n  三个难度分数区分「错误 vs 正确」的 AUROC（0.5=无区分力）')
    print(f'    {"分数":<26}{"AUROC":>8}')
    for name, s in scores.items():
        print(f'    {name:<26}{auroc(s, ~correct):>8.4f}')

    # 高置信错误：easy 流被污染的直接度量
    for thr in (0.9, 0.95):
        m = conf >= thr
        if m.sum():
            print(f'\n  置信度 ≥ {thr}: n={int(m.sum())} ({m.float().mean()*100:.1f}%)，'
                  f'其中错误 {int((~correct[m]).sum())} 个 '
                  f'（错误率 {(~correct[m]).float().mean()*100:.1f}%）')

    print(f'\n  混淆矩阵（行=真值，列=源模型预测）')
    cm = torch.zeros(7, 7, dtype=torch.long)
    for g, p in zip(gt.tolist(), pseudo.tolist()):
        cm[g, p] += 1
    hdr = '真值\\预测'
    print(f'    {hdr:<11}' + ''.join(f'{c[:6]:>8}' for c in PACS_CLASSES) + f'{"召回":>9}')
    for i, c in enumerate(PACS_CLASSES):
        rec = cm[i, i].item() / max(cm[i].sum().item(), 1)
        print(f'    {c:<11}' + ''.join(f'{v:>8}' for v in cm[i].tolist()) + f'{rec*100:>8.1f}%')
    print(f'    {"预测数":<11}' + ''.join(f'{v:>8}' for v in cm.sum(0).tolist()))


if __name__ == '__main__':
    main()
