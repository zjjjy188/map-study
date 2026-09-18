import torch

print("=" * 60)
print("Part 1: Tensor 基础")
print("=" * 60)

# ---------- 1. 创建方式 ----------
print("\n[1] 创建 Tensor")
a = torch.tensor([1, 2, 3])              # 从列表
b = torch.zeros(2, 3)                     # 全 0
c = torch.ones(2, 3)                      # 全 1
d = torch.randn(3, 4)                     # 标准正态随机
e = torch.arange(0, 10, 2)                # 0,2,4,6,8

print("a =", a)
print("b =\n", b)
print("c =\n", c)
print("d =\n", d)
print("e =", e)

# ---------- 2. 关键属性 ----------
print("\n[2] Tensor 属性")
t = torch.randn(32, 3, 224, 224)          # 一个典型图像 batch
print(f"shape      : {t.shape}     # (batch, channel, height, width)")
print(f"ndim       : {t.ndim}      # 维度数")
print(f"dtype      : {t.dtype}     # 数据类型")
print(f"device     : {t.device}    # 在 CPU 还是 GPU")
print(f"numel      : {t.numel()}   # 元素总数 = 32*3*224*224")

# ---------- 3. dtype 和 device ----------
print("\n[3] dtype 和 device")
x = torch.tensor([1.0, 2.0])
print("默认 float dtype:", x.dtype)             # float32

x_int = torch.tensor([1, 2, 3])
print("默认 int dtype  :", x_int.dtype)         # int64

# 移到 GPU
if torch.cuda.is_available():
    x_gpu = x.to("cuda")
    print("移到 GPU 后 device:", x_gpu.device)

# ---------- 4. 形状变换 ----------
print("\n[4] 形状变换")
t = torch.arange(12)
print("原始:", t.shape, t)

t1 = t.reshape(3, 4)#把 12 个元素重新排成 **3 行 4 列** 的二维张量
print("reshape(3,4):\n", t1)

t2 = t1.permute(1, 0)#调换维度顺序（多维转置）
print("permute(1,0) 转置:\n", t2)

t3 = t1.unsqueeze(0)               # 在最前面加一维
print("unsqueeze(0):", t3.shape)

t4 = t1.unsqueeze(0).squeeze(0)    # 再删掉
print("squeeze(0)  :", t4.shape)

# ---------- 5. 运算 ----------
print("\n[5] 运算")
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
print("a =\n", a)
print("b =\n", b)
print("a + b =\n", a + b)
print("a * b =\n", a * b)              # 逐元素
print("a @ b (矩阵乘) =\n", a @ b)

# 广播
print("\n广播机制:")
m = torch.randn(3, 1)
n = torch.randn(1, 4)
print("(3,1) + (1,4) 结果 shape:", (m + n).shape)  # (3,4)