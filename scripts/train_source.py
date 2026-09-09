"""Step 0：源模型训练（research_idea.pdf 图 3 中冻结的「源模型」）。

严格约束：
  * 只读取源域的 train / val 划分
  * 目标域在本脚本中从不出现，模型选择只用源域验证集
  * 目标域准确率由 scripts/eval_source.py 在权重冻结后单独报告

用法:
    python scripts/train_source.py --source art_painting
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from models.source_model import SourceModel, PACS_CLASSES, PACS_DOMAINS, FEATURE_DIM
from utils.pacs_data import PACSSplitDataset, train_transform, eval_transform, evaluate

CKPT_DIR = 'checkpoints/source'


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='art_painting', choices=PACS_DOMAINS)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr-backbone', type=float, default=1e-3)
    ap.add_argument('--lr-head', type=float, default=1e-2)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--label-smoothing', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--smoke', action='store_true',
                    help='只跑几个 batch 验证链路，不产出正式权重')
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f'🖥️  设备: {device} | 源域: {args.source} | 种子: {args.seed}')
    print('⚠️  本脚本不接触任何目标域数据')

    # 只加载 train / val；source test 是留出集，本脚本绝不触碰
    train_ds = PACSSplitDataset(args.source, 'train', train_transform())
    val_ds = PACSSplitDataset(args.source, 'val', eval_transform())
    print(f'📚 源域 train={len(train_ds)}  val={len(val_ds)}')
    print(f'   train 每类: {train_ds.label_counts()}')
    print(f'   val   每类: {val_ds.label_counts()}')

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)

    model = SourceModel().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD([
        {'params': model.backbone.parameters(), 'lr': args.lr_backbone},
        {'params': model.classifier.parameters(), 'lr': args.lr_head},
    ], momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    epochs = 1 if args.smoke else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_epoch = -1.0, -1
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f'{args.source}.pth')

    for epoch in range(epochs):
        model.train()
        run_loss = run_correct = run_n = 0
        t0 = time.time()
        for i, (imgs, labels, _) in enumerate(train_ld):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            run_loss += loss.item() * labels.numel()
            run_correct += (logits.argmax(1) == labels).sum().item()
            run_n += labels.numel()
            if args.smoke and i >= 1:
                print('   [smoke] 已跑 2 个 batch，提前结束')
                break
        scheduler.step()

        val_acc = evaluate(model, val_ld, device)
        print(f'Epoch [{epoch+1}/{epochs}] loss={run_loss/max(run_n,1):.4f} '
              f'train_acc={run_correct/max(run_n,1):.4f} val_acc={val_acc:.4f} '
              f'({time.time()-t0:.0f}s)')

        # 模型选择：只依据源域验证集
        if val_acc > best_acc and not args.smoke:
            best_acc, best_epoch = val_acc, epoch
            model.save(ckpt_path, meta={
                'source_domain': args.source,
                'class_map': PACS_CLASSES,
                'feature_dim': FEATURE_DIM,
                'seed': args.seed,
                'epochs': epochs,
                'best_epoch': best_epoch,
                'src_val_acc': best_acc,
                'selection_criterion': 'source_val_acc_only',
                'label_smoothing': args.label_smoothing,
                'note': '目标域数据未参与训练或模型选择',
            })
            print(f'   💾 新最佳，已保存 {ckpt_path}')

    if args.smoke:
        print('✅ smoke 通过（未保存权重）')
    else:
        print(f'✅ 完成 | 最佳源域验证准确率 {best_acc:.4f} @ epoch {best_epoch+1}')
        print('   源模型训练结束。目标域评测请单独运行 scripts/eval_source.py，'
              '并指定所需的目标域。')


if __name__ == '__main__':
    main()
