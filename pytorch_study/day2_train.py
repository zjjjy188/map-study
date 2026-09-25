import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# ============ 1. 超参数 ============
BATCH_SIZE = 64
EPOCHS = 2
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"使用设备: {DEVICE}")

# ============ 2. 数据准备 ============
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============ 3. 定义模型 ============
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (B,1,28,28) -> (B,16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (B,16,14,14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> (B,32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (B,32,7,7)
            nn.Flatten(),                                 # -> (B,32*7*7)
            nn.Linear(32 * 7 * 7, 10),                    # -> (B,10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN().to(DEVICE)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# ============ 4. 损失函数和优化器 ============
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ============ 5. 训练 ============
def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        # 前向
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 200 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"loss={loss.item():.4f} acc={correct/total:.4f}")

    return total_loss / len(train_loader), correct / total

@torch.no_grad()
def evaluate():
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total

# ============ 6. 开始训练 ============
print("\n开始训练...")
start = time.time()
for epoch in range(1, EPOCHS + 1):
    loss, train_acc = train_one_epoch(epoch)
    test_acc = evaluate()
    print(f"[Epoch {epoch}] loss={loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

print(f"\n训练完成，用时 {time.time()-start:.1f} 秒")
print(f"最终测试准确率: {evaluate():.4f}")

# 保存模型
torch.save(model.state_dict(), "mnist_cnn.pt")
print("模型已保存到 mnist_cnn.pt")