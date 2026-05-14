import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 确保 Python 能找到项目根目录下的 models 和 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.glow_model import SimplifiedGlow
from utils.jacobian_loss import compute_jacobian_alignment

# --- 1. 超参数配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available(): # 针对你的 Mac M1/M2/M3 芯片优化
    DEVICE = torch.device("mps")

EPOCHS = 50
BATCH_SIZE = 16  # 如果显存够大可以改为 32
LR = 1e-4
DOMAIN = 'art_painting' # 目标域
SPLIT_DIM = 8           # 前 8 通道为 Fiber (风格)，后 4 通道为 Base (语义)
SAVE_PATH = "checkpoints/glow_stage1.pth"

# --- 2. 加载 PACS 数据集 ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 路径指向你上传截图中显示的 data 目录
data_root = os.path.join('data', DOMAIN) 
if not os.path.exists(data_root):
    # 兼容性处理：如果 data 文件夹下还有一层 PACS 文件夹
    data_root = os.path.join('data', 'PACS', DOMAIN)

print(f"正在加载数据自: {data_root}")
dataset = datasets.ImageFolder(root=data_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # Mac上num_workers设为0更稳定

# --- 3. 初始化模型与优化器 ---
model = SimplifiedGlow(num_layers=12).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

print(f"🚀 开始在 {DOMAIN} 领域训练 Glow... 使用设备: {DEVICE}")

# --- 4. 训练主循环 ---
model.train()
for epoch in range(EPOCHS):
    total_nll = 0
    total_ja = 0
    
    for i, (images, _) in enumerate(dataloader):
        images = images.to(DEVICE)
        
        # A. 计算负对数似然 (NLL)
        # NLL = -log p(x) = -(log p(z) + log |det(J)|)
        z, log_det = model.transform_to_noise(images)
        
        # 假设 z 服从标准正态分布 N(0, I)
        log_p_z = -0.5 * torch.sum(z**2, dim=[1, 2, 3])
        loss_nll = -(log_p_z + log_det).mean()
        
        # B. 雅可比对齐损失 (JA Loss) - 几何解耦的核心
        # 为了稳定，前 5 个 epoch 只训练重构，之后开启解耦约束
        if epoch >= 5:
            # compute_jacobian_alignment 已经在内部做了扰动近似
            loss_ja = compute_jacobian_alignment(model, images, split_dim=SPLIT_DIM)
            loss = loss_nll + 0.5 * loss_ja # 这里的权重 0.5 可以根据 JA 下降情况微调
        else:
            loss_ja = torch.tensor(0.0).to(DEVICE)
            loss = loss_nll
            
        # C. 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸 (INN 训练常用技巧)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_nll += loss_nll.item()
        total_ja += loss_ja.item()
        
        if i % 10 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Step [{i}/{len(dataloader)}] | "
                  f"NLL: {loss_nll.item():.4f} | JA: {loss_ja.item():.4f}")

    # 保存模型权重
    torch.save(model.state_dict(), SAVE_PATH)
    avg_nll = total_nll / len(dataloader)
    print(f"--- Epoch {epoch} 完成 | 平均 NLL: {avg_nll:.4f} | 模型已更新 ---")

print("✅ Stage 1 训练圆满结束！")