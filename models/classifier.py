import torchvision.models as models
import torch.nn as nn

def build_classifier(num_classes):
    # 使用 ResNet50 作为基准骨架
    model = models.resnet50(pretrained=True)
    # 替换最后的全连接层以适应你的类别数
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model