"""PACS 数据读取：统一走 splits/ 下的确定性划分清单。"""
import os
import sys

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

sys.path.append(os.getcwd())
from scripts.prepare_pacs import find_data_root, list_domain, load_split

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(size=224):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.05),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size=224):
    """确定性变换：评测与特征缓存都用它。"""
    return T.Compose([
        T.Resize(int(size * 256 / 224)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class PACSSplitDataset(Dataset):
    """按 splits/{domain}_{split}.txt 读取；split='all' 时读取整个域。

    __getitem__ 返回 (image, label, relative_path)。
    """

    def __init__(self, domain, split, transform, root=None):
        self.root = root or find_data_root()
        if self.root is None:
            raise FileNotFoundError(
                '找不到 PACS 数据，请先运行 scripts/prepare_pacs.py')
        if split == 'all':
            self.items, _ = list_domain(self.root, domain)
        else:
            self.items = load_split(domain, split)
        self.domain = domain
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel, label = self.items[idx]
        img = Image.open(os.path.join(self.root, rel)).convert('RGB')
        return self.transform(img), label, rel

    def label_counts(self):
        c = {}
        for _, l in self.items:
            c[l] = c.get(l, 0) + 1
        return [c.get(i, 0) for i in range(7)]


@torch.no_grad()
def evaluate(model, loader, device):
    """返回 top-1 准确率。model 需已 eval()。"""
    model.eval()
    correct = total = 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        pred = model(imgs).argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)
