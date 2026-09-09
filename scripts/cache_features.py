"""Step 0：目标域特征缓存。

F 冻结后目标域特征是确定的，一次算好可让后续 Stage 1 完全不必重跑 ResNet。
使用确定性的 eval_transform（不做增强），保证缓存可复现。

⚠️ 真值标签的隔离约定
    真值只用于评测记账，绝不可进入适应流程。因此它被单独放在 'eval_only'
    子字典中，并且 load_target_cache() 默认不返回该键 —— 适应阶段的代码
    即使写错也拿不到它。要取真值必须显式传 include_eval_only=True。

用法:
    python scripts/cache_features.py --source art_painting --target photo
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from models.source_model import SourceModel, PACS_DOMAINS
from utils.pacs_data import PACSSplitDataset, eval_transform
from scripts.train_source import get_device, CKPT_DIR

CACHE_DIR = 'cache/features'
EVAL_ONLY_KEY = 'eval_only'


def load_target_cache(path, include_eval_only=False):
    """读取缓存。默认剥离 eval_only（真值），防止泄漏进适应流程。"""
    d = torch.load(path, map_location='cpu')
    if not include_eval_only:
        d.pop(EVAL_ONLY_KEY, None)
    return d


@torch.no_grad()
def build_cache(source, target, batch_size=64, workers=4):
    device = get_device()
    ckpt_path = os.path.join(CKPT_DIR, f'{source}.pth')
    model, meta = SourceModel.load(ckpt_path, device=device)
    model.freeze()

    ds = PACSSplitDataset(target, 'all', eval_transform())
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    print(f'🖥️  设备: {device} | 目标域 {target}: {len(ds)} 张')

    feats, logits_all, paths, gts = [], [], [], []
    for imgs, labels, rels in ld:
        f = model.features(imgs.to(device))
        lg = model.classify(f)
        feats.append(f.cpu())
        logits_all.append(lg.cpu())
        paths.extend(rels)
        gts.append(labels)

    feature = torch.cat(feats).float()
    logits = torch.cat(logits_all).float()
    prob = logits.softmax(1)
    entropy = -(prob * prob.clamp_min(1e-12).log()).sum(1)
    pseudo = prob.argmax(1)

    payload = {
        'meta': {
            'source_domain': source, 'target_domain': target,
            'feature_dim': feature.shape[1], 'num_samples': feature.shape[0],
            'transform': 'eval_transform(224), 无增强',
            'checkpoint': ckpt_path, 'checkpoint_meta': meta,
            'entropy_max': float(torch.tensor(prob.shape[1]).float().log()),
        },
        # ---- 适应阶段可用 ----
        'feature': feature,             # [N, 2048]
        'logits': logits,               # [N, 7]
        'prob': prob,                   # [N, 7]
        'entropy': entropy,             # [N]
        'pseudo_label': pseudo,         # [N]
        'path': paths,                  # [N]
        # ---- 仅供评测记账，load_target_cache 默认不返回 ----
        EVAL_ONLY_KEY: {
            'label': torch.cat(gts),
            'note': '真值仅用于评测；严禁作为适应阶段的输入或模型选择依据',
        },
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    out = os.path.join(CACHE_DIR, f'{source}__{target}.pt')
    torch.save(payload, out)

    ent = entropy
    print(f'💾 已写入 {out}  ({os.path.getsize(out)/1e6:.1f} MB)')
    print(f'   feature {tuple(feature.shape)}  entropy: '
          f'min={ent.min():.3f} 中位={ent.median():.3f} max={ent.max():.3f} '
          f'(上界 ln7={payload["meta"]["entropy_max"]:.3f})')
    print(f'   伪标签分布: {torch.bincount(pseudo, minlength=7).tolist()}')
    print(f'   🔒 真值已隔离在 "{EVAL_ONLY_KEY}"，load_target_cache() 默认不返回')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='art_painting', choices=PACS_DOMAINS)
    ap.add_argument('--target', default='photo', choices=PACS_DOMAINS)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    assert a.source != a.target, '源域与目标域不能相同'
    build_cache(a.source, a.target, a.batch_size, a.workers)
