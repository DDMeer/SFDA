import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from models.glow_model import SimplifiedGlow

# --- 配置参数 ---
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 100 # 增加到 100 轮
DOMAIN = 'art_painting'
DATA_ROOT = f"data/{DOMAIN}" # 确保指向你的图片文件夹

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 数据加载
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(root=DATA_ROOT, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SimplifiedGlow().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"🚀 开始训练... 语义划分: 6:6 | JA 权重: 1.0")

for epoch in range(EPOCHS):
    model.train()
    for i, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device)
        
        # 得到解耦后的向量
        z_s, z_c, log_det = model.transform_to_noise(imgs)
        
        # 1. 计算重构损失 (NLL)
        # 假设隐空间服从标准正态分布
        z = torch.cat([z_s, z_c], dim=1)
        nll_loss = 0.5 * torch.sum(z**2) - log_det
        nll_loss = nll_loss / imgs.size(0)
        
        # 2. 计算解耦损失 (JA Loss)
        # 只有在 Epoch > 5 时开启，强制让 z_s 的变化不影响 z_c
        loss_ja = torch.tensor(0.0).to(device)
        if epoch >= 5:
            z_s.requires_grad_(True)
            # 简单的雅可比近似：计算 z_c 对 z_s 的梯度
            grad = torch.autograd.grad(outputs=z_c.sum(), inputs=z_s, create_graph=True)[0]
            loss_ja = torch.norm(grad)
        
        # --- 核心修改：提升 JA 权重至 1.0 ---
        total_loss = nll_loss + 1.0 * loss_ja
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Step [{i}/{len(loader)}] | NLL: {nll_loss.item():.4f} | JA: {loss_ja.item():.4f}")

    # 保存模型
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/glow_stage1.pth')