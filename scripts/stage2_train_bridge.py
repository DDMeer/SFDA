import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import clip
import os
import sys

# 确保导入 SimplifiedGlow
sys.path.append(os.getcwd())
from models.glow_model import SimplifiedGlow

# --- 1. 定义映射网络 (符合你的 Base=Concept 理论) ---
class ConceptBridge(nn.Module):
    def __init__(self, zc_channels=8, img_size=32, clip_dim=512):
        super().__init__()
        # 输入是底座 z_c: [Batch, 8, 32, 32]
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(zc_channels * img_size * img_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, clip_dim) 
        )

    def forward(self, z_c):
        return self.net(z_c)

# --- 2. 环境配置 ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 16 # M2 Max 建议 16-32
LR = 1e-4
EPOCHS = 30
DOMAIN = 'art_painting'
DATA_ROOT = os.path.join("data", DOMAIN)

# 加载 CLIP (语义金标准)
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# 加载你刚训练好的 Stage 1 Glow
glow_model = SimplifiedGlow().to(device)
checkpoint_path = 'checkpoints/glow_stage1.pth'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError("找不到 Stage 1 权重，请先完成 Glow 训练！")
glow_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
glow_model.eval() # 锁定 Glow

# 初始化映射网络
bridge_net = ConceptBridge(zc_channels=8).to(device) # 对应 4:8 划分
optimizer = optim.Adam(bridge_net.parameters(), lr=LR)
criterion = nn.CosineEmbeddingLoss()

# --- 3. 数据准备 ---
dataset = datasets.ImageFolder(root=DATA_ROOT, transform=transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
]))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 准备类别的文本描述
class_names = dataset.classes
text_prompts = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)

print(f"🔗 Stage 2 启动: 正在将几何底座 z_c 映射至 CLIP 文本空间")
print(f"💻 运行设备: {device} | 类别: {class_names}")

# --- 4. 训练循环 ---
for epoch in range(EPOCHS):
    bridge_net.train()
    epoch_loss = 0
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        with torch.no_grad():
            # 提取底座 z_c
            _, z_c, _ = glow_model.transform_to_noise(imgs)
            # 获取 CLIP 文本特征
            text_features = clip_model.encode_text(text_prompts)[labels]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 预测语义向量
        pred_features = bridge_net(z_c)
        pred_features = pred_features / pred_features.norm(dim=-1, keepdim=True)
        
        # 计算余弦相似度损失
        loss = criterion(pred_features, text_features, torch.ones(imgs.size(0)).to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    print(f"Epoch [{epoch}/{EPOCHS}] | Loss: {epoch_loss/len(loader):.4f}")
    
    # 保存映射模型
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(bridge_net.state_dict(), 'checkpoints/bridge_stage2.pth')

print("✅ Stage 2 训练完成！底座现在具备语义了。")