"""源模型 (Source Model)：research_idea.pdf 图 3 中冻结的「源模型」。

提供两个后续阶段唯一依赖的接口：
    F(x) -> [B, 2048]   特征提取器（可逆网络的输入）
    C(f) -> [B, 7]      分类头（用于难易划分与 L_sem 的语义一致性）
"""
import torch
import torch.nn as nn
import torchvision.models as tvm

PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
PACS_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
FEATURE_DIM = 2048


class SourceModel(nn.Module):
    """ResNet-50 骨干 + 线性分类头，拆成 F / C 两段显式暴露。"""

    def __init__(self, num_classes=len(PACS_CLASSES), pretrained=True):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.resnet50(weights=weights)
        # F: conv1 ... layer4 -> avgpool -> flatten
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        # C: 2048 -> num_classes
        self.classifier = nn.Linear(FEATURE_DIM, num_classes)
        self.feature_dim = FEATURE_DIM
        self.num_classes = num_classes

    # ---- 后续阶段的公开接口 ----
    def features(self, x):
        """F(x) -> [B, 2048]"""
        return torch.flatten(self.backbone(x), 1)

    def classify(self, f):
        """C(f) -> [B, num_classes]"""
        return self.classifier(f)

    def forward(self, x):
        return self.classify(self.features(x))

    def freeze(self):
        """冻结 F 与 C（Stage 1 起两者都不再更新）。"""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    # ---- 持久化 ----
    def save(self, path, meta):
        torch.save({
            'backbone': self.backbone.state_dict(),
            'classifier': self.classifier.state_dict(),
            'meta': meta,
        }, path)

    @classmethod
    def load(cls, path, device='cpu', num_classes=len(PACS_CLASSES)):
        ckpt = torch.load(path, map_location=device)
        model = cls(num_classes=num_classes, pretrained=False)
        model.backbone.load_state_dict(ckpt['backbone'])
        model.classifier.load_state_dict(ckpt['classifier'])
        return model.to(device), ckpt['meta']
