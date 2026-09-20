"""
exp01: CNN 层数与学习率对比实验
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 2

# ============ 数据 ============
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ============ 模型定义：可配置层数 ============
class CNN(nn.Module):
    def __init__(self, n_conv=2):
        """
        n_conv: 2 或 4，表示卷积层数
        """
        super().__init__()
        layers = []

        if n_conv == 2:
            # 2 层卷积
            layers += [
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            ]
            flat_dim = 32 * 7 * 7
        elif n_conv == 4:
            # 4 层卷积
            layers += [
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            ]
            flat_dim = 32 * 7 * 7

        layers += [nn.Flatten(), nn.Linear(flat_dim, 10)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============ 训练与评估 ============
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ============ 主实验循环 ============
def run_experiment(name, n_conv, lr):
    print(f"\n{'='*60}")
    print(f"实验 {name}: n_conv={n_conv}, lr={lr}")
    print(f"{'='*60}")

    model = CNN(n_conv=n_conv).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    history = []
    for epoch in range(1, EPOCHS + 1):
        loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_acc = evaluate(model, test_loader)
        print(f"  Epoch {epoch}: loss={loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
        history.append({
            "epoch": epoch,
            "loss": round(loss, 4),
            "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4),
        })

    elapsed = time.time() - start
    result = {
        "name": name,
        "n_conv": n_conv,
        "lr": lr,
        "params": sum(p.numel() for p in model.parameters()),
        "elapsed_sec": round(elapsed, 1),
        "final_test_acc": history[-1]["test_acc"],
        "history": history,
    }
    print(f"  用时 {elapsed:.1f} 秒，最终 test_acc={history[-1]['test_acc']:.4f}")
    return result


if __name__ == "__main__":
    configs = [
        ("A_基线",       2, 0.001),
        ("B_深层",       4, 0.001),
        ("C_大学习率",   2, 0.01),
        ("D_小学习率",   2, 0.0001),
    ]
    results = []
    for name, n_conv, lr in configs:
        results.append(run_experiment(name, n_conv, lr))

    # 保存结果
    with open("exp01_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("实验汇总表")
    print("=" * 60)
    print(f"{'实验':<12}{'层数':<6}{'lr':<10}{'参数量':<12}{'用时(s)':<10}{'test_acc':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<12}{r['n_conv']:<6}{r['lr']:<10}"
              f"{r['params']:<12}{r['elapsed_sec']:<10}{r['final_test_acc']:<10.4f}")
    print("=" * 60)
    print("结果已保存到 exp01_results.json")