import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
import os
import sys
import clip

# 1. 环境补丁 (针对 Mac MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

sys.path.append(os.getcwd())
from models.glow_model import SimplifiedGlow

# 2. 映射网络架构 (严格对齐 Stage 3 的 9 层结构)
class ConceptBridge(nn.Module):
    def __init__(self, zc_channels=8, img_size=32, clip_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                                       # 0
            nn.Linear(zc_channels * img_size * img_size, 1024), # 1
            nn.BatchNorm1d(1024),                               # 2
            nn.ReLU(),                                          # 3
            nn.Dropout(0.3),                                    # 4
            nn.Linear(1024, 512),                               # 5
            nn.BatchNorm1d(512),                                # 6
            nn.ReLU(),                                          # 7
            nn.Linear(512, clip_dim)                            # 8
        )
    def forward(self, z_c): return self.net(z_c)

# 3. 数据集加载 (适配 PACS art_painting)
class PACSDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        # CLASS_MAP = {0: "dog", 1: "elephant", 2: "giraffe", 3: "guitar", 4: "horse", 5: "house", 6: "person"}
        for label in range(7):
            class_dir = os.path.join(root_dir, str(label))
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(label)
        
        self.transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]

# 4. 训练主程序
def train_bridge():
    # A. 加载 Stage 1 模型 (用于提取 z_c)
    glow = SimplifiedGlow().to(device)
    glow.load_state_dict(torch.load('checkpoints/glow_stage1.pth', map_location=device))
    glow.eval()

    # B. 加载 CLIP 模型 (获取语义目标)
    clip_model, _ = clip.load("ViT-B/32", device=device)
    class_names = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
    with torch.no_grad():
        text_tokens = clip.tokenize([f"a photo of a {c}" for c in class_names]).to(device)
        target_clip_features = clip_model.encode_text(text_tokens).float()
        target_clip_features /= target_clip_features.norm(dim=-1, keepdim=True)

    # C. 初始化 Bridge 网络
    bridge = ConceptBridge(zc_channels=8).to(device) # 使用 8 通道模式
    optimizer = optim.Adam(bridge.parameters(), lr=1e-4)
    criterion = nn.MSELoss() # 目标是让输出向量靠近 CLIP 目标向量

    # D. 加载数据 (使用艺术画域名作为源域训练)
    dataset = PACSDataset("data/art_painting")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"🚀 开始重训 Stage 2 | 数据量: {len(dataset)} | 设备: {device}")

    epochs = 50
    for epoch in range(epochs):
        bridge.train()
        total_loss = 0
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                # 1. 显式解包：transform_to_noise 返回 (z_s, z_c, log_det)
                z_s, z_c, _ = glow.transform_to_noise(imgs)
                assert z_s.shape[1] == 4, f"z_s 应为 4 通道 (style)，实得 {z_s.shape[1]}"
                assert z_c.shape[1] == 8, f"z_c 应为 8 通道 (content)，实得 {z_c.shape[1]}"

                # 2. z_c 本身即 8 通道，直接作为 Bridge 输入（不再需要 padding 补齐）
                z_c_input = z_c # [B, 8, 32, 32]

            # 3. 前向传播
            pred_features = bridge(z_c_input)
            pred_features = pred_features / pred_features.norm(dim=-1, keepdim=True)
            
            # 4. 获取对应的 CLIP 目标特征
            target_features = target_clip_features[labels]

            loss = criterion(pred_features, target_features)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(dataloader):.6f}")

    # E. 保存模型
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(bridge.state_dict(), 'checkpoints/bridge_stage2.pth')
    print("✅ Stage 2 重训完成并保存为 checkpoints/bridge_stage2.pth")

if __name__ == "__main__":
    train_bridge()