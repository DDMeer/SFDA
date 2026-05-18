import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys

sys.path.append(os.getcwd())
from models.glow_model import SimplifiedGlow

# --- 配置参数 ---
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 100       
DOMAIN = 'art_painting'
DATA_ROOT = os.path.join("data", DOMAIN) 
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- 数据加载 ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
if not os.path.exists(DATA_ROOT):
    raise FileNotFoundError(f"找不到数据集目录: {DATA_ROOT}")
dataset = datasets.ImageFolder(root=DATA_ROOT, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 模型初始化 ---
model = SimplifiedGlow().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"🚀 启动 Stage 1 暴力解耦模式...")
print(f"📐 隐空间划分: 4(Style):8(Content) | JA 权重: 5.0")

for epoch in range(EPOCHS):
    model.train()
    epoch_nll = 0
    epoch_ja = 0
    
    for i, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device)
        
        # 开启对输入的梯度追踪
        if epoch >= 5:
            imgs.requires_grad_(True)
        
        z_s, z_c, log_det = model.transform_to_noise(imgs)
        
        # A. 重构损失 (NLL)
        z_total = torch.cat([z_s, z_c], dim=1)
        nll_loss = (0.5 * torch.sum(z_total**2) - torch.sum(log_det)) / imgs.size(0)
        
        # B. 暴力解耦约束 (JA Loss)
        loss_ja = torch.tensor(0.0).to(device)
        if epoch >= 5:
            grad_c = torch.autograd.grad(outputs=z_c.sum(), inputs=imgs, create_graph=True, retain_graph=True)[0]
            grad_s = torch.autograd.grad(outputs=z_s.sum(), inputs=imgs, create_graph=True, retain_graph=True)[0]
            # 强制梯度正交：点积趋近于 0
            loss_ja = torch.abs(torch.sum(grad_c * grad_s)) / imgs.size(0)
        
        # --- 核心修改：赋予解耦极高的优先级 ---
        total_loss = nll_loss + 5.0 * loss_ja
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_nll += nll_loss.item()
        epoch_ja += loss_ja.item()
        
        if i % 20 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Step [{i}/{len(loader)}] | NLL: {nll_loss.item():.4f} | JA: {loss_ja.item():.4f}")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/glow_stage1.pth')
    print(f"--- Epoch {epoch} 完成 | 平均 NLL: {epoch_nll/len(loader):.4f} | 平均 JA: {epoch_ja/len(loader):.4f} ---")