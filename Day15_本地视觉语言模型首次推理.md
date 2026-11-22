# Day15｜本地视觉语言模型首次推理

日期：2026-11-22。难度中等。本日目标为运行单张地图推理，保存并核查实际输出。

## 1. 实际完成

- 下载 AGI-GIS/MapReader，版本为 `4d6d4598cb3fbbb9376ff5bd6c62fde344c2dd23`，最终位置为 `D:\project\map-study\MapReader`。
- 核验本机 GPU 与 PyTorch CUDA 环境。
- 打开官方地图 `data/MapEval-DB/Country/China/Map/12_16.jpg`（256×256像素）及对应参考描述。
- 因显存与权重获取条件，转为先运行 SmolVLM-500M-Instruct，完成图片输入、GPU加载、文本生成与人工核查。MapReader原模型尚未运行。
- 脚本见 [day15_smolvlm.py](pytorch_study/day15_smolvlm.py)，两次输出记录见 [model_results.json](runs/day15/model_results.json)。

## 2. 环境与模型

- Windows；Python 3.10.21，环境为 `D:\miniconda\envs\map-study`。
- RTX 4060 Laptop，显存8188MiB；驱动560.94。
- PyTorch `2.6.0+cu124`，其CUDA版本12.4，GPU可用；Transformers `4.57.6`，Accelerate `1.15.0`。
- `nvidia-smi`显示的CUDA 12.6是驱动支持的版本上限；PyTorch实际使用的版本另行检查。
- 模型为 `HuggingFaceTB/SmolVLM-500M-Instruct`，本地目录 `D:\AI-models\SmolVLM-500M-Instruct`。
- 权重下载元数据记录的修订为 `753c1752602b5e739af7a7fa6947a635051897a3`。使用FP16、SDPA、单张图片，模型加载后观测到PyTorch已分配显存约988MiB；这不是推理峰值或整机总占用。

## 3. 从操作中学到的知识

- **CMD与PowerShell**是不同的命令解释器。当前使用CMD，列目录用`dir`，跨盘切换用`cd /d 路径`。
- **GPU**是计算硬件，**CUDA**提供NVIDIA GPU计算支持，**PyTorch**组织模型计算。CUDA可用后还需检查显存是否足够。
- **Transformers**加载模型、预处理输入和生成文字；**Accelerate**辅助加载及设备管理。
- **权重**是训练得到的参数。下载代码、安装工具、下载权重是不同步骤。
- 脚本数据流为“读取图片与问题 → processor转换输入 → 模型generate → 解码成文字”。
- `.to('cuda')`将模型或输入移到GPU；FP16降低存储占用；`.eval()`设置评估模式；`torch.inference_mode()`关闭推理时不需要的自动求导记录。
- `max_new_tokens=100`限制新生成token数；`do_sample=False`采用非随机采样方式，但不据此保证跨平台逐字一致。

## 4. 两次实际结果

### 宽泛描述

问题为“Describe the main features visible in this map.”。

模型生成了地图描述，但把沙咀村名称读成`SHAZICUCN`并归为街道，还声称阳逻居民区由蓝线标示，图片缺少相应支持。回答在句中结束，可能受到100个新token的上限影响，未单独检验该原因。

### 具体问答

仅修改脚本中的问题为“Does the G70 highway cross the river in this map? Answer briefly.”，用户反馈模型回答`yes`。该回答与图中G70跨河的情况一致。本条根据用户反馈记录，未取得第二次完整终端文本。

两次任务分别是开放描述和具体问答，可观察不同任务的表现；单一样例不足以证明提示词优化或整体能力提高。

## 5. 为什么要核查参考描述

官方参考描述声称长江从西北流向东南，但当前图片未见流向箭头等直接证据。河道延伸方向不能直接证明水流方向，外部地理知识与图像证据需区分。以后分析模型答案时，也应明确每条判断的依据。

## 6. 运行与常用命令

先在CMD执行 `cd /d D:\project\map-study\MapReader`，切到实际运行目录，再执行：

```cmd
D:\miniconda\envs\map-study\python.exe ..\pytorch_study\day15_smolvlm.py
```

该命令使用指定环境运行脚本；`..`表示上一级目录。脚本中的模型路径、地图路径为本机绝对路径，换机器需修改。

- **建议记住**：`nvidia-smi`查看本机GPU状态；`python -m pip install 包名`使用指定Python环境安装包。
- `start "" "文件路径"`用默认程序打开文件；`notepad 文件路径`编辑文本。
- `Ctrl+C`在终端中中断运行，已完成的下载文件会保留；`Ctrl+S`在编辑器中保存。
- 最初整仓库下载额外获取了ONNX格式文件，后清理约5.48GB。本次使用的约968MiB权重与配置保留在D盘。

## 7. 当前进度与下一步

已实现本机小型视觉语言模型的首次推理。MapReader官方默认脚本为8卡配置，CogVLM官方所列INT4推理约需11GB显存，不能直接照搬到本机8GB显卡；其他适配方案尚未验证。未在已检查的MapReader仓库及说明中找到微调后权重的明确下载入口，已询问老师GPU资源，等待回复。

下一步拟选择少量地图，固定问题与生成参数，保存原始回答并分析错误。服务器和权重条件明确后再推进MapReader原模型。

来源：[MapReader官方仓库](https://github.com/AGI-GIS/MapReader)、[CogVLM硬件要求](https://github.com/zai-org/CogVLM#hardware-requirement)、[SmolVLM模型说明](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)。
