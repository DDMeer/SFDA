"""Step 0 评测：在权重冻结后报告三组基线数字（论文 baseline 表）。

  1. 源域验证准确率      —— 合理性检查（模型选择用过）
  2. 源域留出集准确率    —— 抗遗忘基准线；该划分从未参与训练与模型选择
  3. 目标域适应前准确率  —— SFDA 论文里的 "source-only" 行

本脚本只读取 checkpoint，不做任何训练或模型选择。

用法:
    python scripts/eval_source.py --source art_painting --target photo
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from models.source_model import SourceModel, PACS_DOMAINS
from utils.pacs_data import PACSSplitDataset, eval_transform, evaluate
from scripts.train_source import get_device, CKPT_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='art_painting', choices=PACS_DOMAINS)
    ap.add_argument('--target', default='photo', choices=PACS_DOMAINS)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    assert args.source != args.target, '源域与目标域不能相同'

    device = get_device()
    ckpt_path = os.path.join(CKPT_DIR, f'{args.source}.pth')
    model, meta = SourceModel.load(ckpt_path, device=device)
    model.freeze()
    print(f'🖥️  设备: {device}')
    print(f'📦 checkpoint: {ckpt_path}')
    print(f'   训练时源域验证准确率: {meta["src_val_acc"]:.4f} (epoch {meta["best_epoch"]+1})')
    print(f'   模型选择依据: {meta["selection_criterion"]}\n')

    def acc(domain, split):
        ds = PACSSplitDataset(domain, split, eval_transform())
        ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
        return evaluate(model, ld, device), len(ds)

    src_val, n1 = acc(args.source, 'val')
    src_test, n2 = acc(args.source, 'test')     # 留出集：未参与训练与模型选择
    tgt_all, n3 = acc(args.target, 'all')       # 目标域真值仅用于评测

    print('📊 适应前基线')
    print(f'   源域验证   ({args.source}/val,  n={n1})  : {src_val:.4f}   （模型选择用过）')
    print(f'   源域留出   ({args.source}/test, n={n2})  : {src_test:.4f}   ← 抗遗忘基准线')
    print(f'   目标域     ({args.target}/all,  n={n3})  : {tgt_all:.4f}   ← source-only 基线')
    print('\n   抗遗忘定义: Forgetting = src_test_before − src_test_after')

    out = {
        'source_domain': args.source, 'target_domain': args.target,
        'src_val_acc': src_val,
        'src_test_acc_before': src_test,
        'tgt_before_adaptation_acc': tgt_all,
        'n_src_val': n1, 'n_src_test': n2, 'n_tgt_all': n3,
        'protocol': {
            'anti_forgetting_baseline': 'source test split（未参与训练与模型选择）',
            'source_only_baseline': 'target 全量，仅评测使用真值',
        },
        'checkpoint_meta': meta,
    }
    os.makedirs('results/baselines', exist_ok=True)
    p = f'results/baselines/{args.source}__{args.target}_before.json'
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f'\n✅ 已写入 {p}')


if __name__ == '__main__':
    main()
