import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

print("=" * 60)
print("Part 2: Dataset + DataLoader")
print("=" * 60)

# ---------- 1. 定义 transforms ----------
transform = transforms.Compose([
    transforms.ToTensor(),                          # PIL Image -> Tensor [0,1]
    transforms.Normalize((0.1307,), (0.3081,)),    # MNIST 均值/标准差标准化
])

# ---------- 2. 下载并加载数据集 ----------
print("\n[1] 加载 MNIST 数据集（首次会下载）")
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# ---------- 3. 看一个样本 ----------
print("\n[2] 看第一个样本")
img, label = train_dataset[0]
print(f"单张图像 shape : {img.shape}    # (channel, height, width)")
print(f"单张图像 dtype : {img.dtype}")
print(f"标签           : {label}        # 是数字 0-9")

# ---------- 4. DataLoader 分批 ----------
print("\n[3] DataLoader 分批")
BATCH_SIZE = 32
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,      # Windows 上先用 0，避免多进程报错
)

print(f"batch_size = {BATCH_SIZE}")
print(f"总 batch 数 = {len(train_loader)}")

# ---------- 5. 取一个 batch ----------
print("\n[4] 取一个 batch 看看")
imgs, labels = next(iter(train_loader))
print(f"一个 batch 的 imgs shape   : {imgs.shape}     # (B, C, H, W)")
print(f"一个 batch 的 labels shape : {labels.shape}")
print(f"labels 前 10 个: {labels[:10].tolist()}")

# ---------- 6. 可视化几张图 ----------
print("\n[5] 保存前 6 张图预览")
fig, axes = plt.subplots(2, 3, figsize=(8, 5))
for i, ax in enumerate(axes.flat):
    # 从 (1, 28, 28) 变成 (28, 28)
    img_np = imgs[i].squeeze().numpy()
    ax.imshow(img_np, cmap="gray")
    ax.set_title(f"label={labels[i].item()}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("batch_preview.png", dpi=100)
print("已保存 batch_preview.png")

print("\n" + "=" * 60)
print("Day2 DataLoader 测试完成！")
print("=" * 60)