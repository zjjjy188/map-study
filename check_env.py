import sys
import platform
import torch

line = "=" * 60

def safe(fn, default="N/A"):
    try:
        return fn()
    except Exception as e:
        return f"<错误: {e}>"

print(line)
print("系统环境检查报告")
print(line)

# ---------- 1. 系统信息 ----------
print("\n[1] 系统信息")
print(f"操作系统      : {platform.system()} {platform.release()} ({platform.version()})")
print(f"机器架构      : {platform.machine()}")
print(f"处理器        : {platform.processor()}")
print(f"Python 版本   : {sys.version.split()[0]}")
print(f"Python 路径   : {sys.executable}")

# ---------- 2. PyTorch 信息 ----------
print("\n[2] PyTorch 信息")
print(f"PyTorch 版本  : {torch.__version__}")
print(f"编译时 CUDA   : {torch.version.cuda}")
print(f"cuDNN 版本    : {torch.backends.cudnn.version()}")
print(f"cuDNN 是否启用: {torch.backends.cudnn.enabled}")
print(f"MKL 是否可用  : {safe(lambda: torch.backends.mkl.is_available())}")

# ---------- 3. CUDA / GPU 信息 ----------
print("\n[3] CUDA / GPU 信息")
cuda_ok = torch.cuda.is_available()
print(f"CUDA 是否可用 : {cuda_ok}")

if not cuda_ok:
    print("  -> 当前使用 CPU。可能原因：")
    print("     - 安装的是 CPU 版 PyTorch")
    print("     - 显卡驱动过旧")
    print("     - 无 NVIDIA 显卡")
else:
    print(f"GPU 数量      : {torch.cuda.device_count()}")
    print(f"当前 GPU 索引 : {torch.cuda.current_device()}")

    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved  = torch.cuda.memory_reserved(i) / (1024 ** 3)
        cc_major, cc_minor = torch.cuda.get_device_capability(i)

        print(f"\n  --- GPU {i} ---")
        print(f"  名称        : {name}")
        print(f"  总显存      : {total_mem:.2f} GB")
        print(f"  已分配显存  : {allocated:.4f} GB")
        print(f"  已缓存显存  : {reserved:.4f} GB")
        print(f"  计算能力    : {cc_major}.{cc_minor}")
        print(f"  多处理器数  : {props.multi_processor_count}")

# ---------- 4. 张量运算测试 ----------
print("\n[4] 张量运算测试")

cpu_t = torch.rand(3, 3)
print("CPU 张量:\n", cpu_t)

if cuda_ok:
    try:
        gpu_t = torch.rand(3, 3, device="cuda")
        result = gpu_t @ gpu_t.T
        torch.cuda.synchronize()
        print("\nGPU 张量（矩阵乘法结果）:\n", result)
        print("GPU 计算成功 [OK]")
        print(f"计算后已分配显存: {torch.cuda.memory_allocated(0)/(1024**3):.4f} GB")
        del gpu_t, result
        torch.cuda.empty_cache()
        print("已清空 GPU 缓存")
    except Exception as e:
        print(f"GPU 计算失败 [X]: {e}")
else:
    print("\n跳过 GPU 测试（CUDA 不可用）")

# ---------- 5. 汇总 ----------
print("\n" + line)
print("检查结论")
print(line)
print("CPU 可用        : [OK]")
print(f"GPU (CUDA) 可用 : {'[OK]' if cuda_ok else '[X]'}")
if cuda_ok:
    print(f"主力 GPU        : {torch.cuda.get_device_name(0)}")
    print(f"显存            : {torch.cuda.get_device_properties(0).total_memory/(1024**3):.2f} GB")
print(line)