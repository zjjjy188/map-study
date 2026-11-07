## ***day1*** 

做了：Conda 环境管理：create / activate / env list / conda init

PyTorch GPU 安装：--index-url .../cu124

环境验证：torch.cuda.is\_available() + get\_device\_name() + cuDNN

Windows 路径坑：cd /d 跨盘符、不能有中文/空格

Git 基础：init / add / commit / remote / push

SSH 密钥：ssh-keygen + GitHub SSH keys + ssh -T 验证

HTTPS → SSH 切换：git remote set-url

加油，第一天开始，虽然啥也不懂但好歹也是开始了。先下跑步了。明天再研究一下怎么上传文件吧。

明天打开电脑，只需要：

cmd

:: ① 打开 Anaconda Prompt (miniconda3)

:: ② 切到项目目录 + 激活环境

cd /d D:\\project\\map-study

conda activate map-study

:: ③ 确认没问题

python -c "import torch; print(torch.cuda.is\_available())"

看到 True 就可以开始学习了。

## ***day2*** 
tensor:
<img width="1747" height="893" alt="image" src="https://github.com/user-attachments/assets/b6fc230e-f621-4269-8355-29266b91d17b" />
shape = (32, 3, 224, 224) 分别代表什么？
torch.randn 和 torch.rand 区别是啥？
为什么要 .to("cuda")？
<img width="1237" height="917" alt="image" src="https://github.com/user-attachments/assets/b2ea4c04-4a18-4aea-b6bc-fff2c5093a9d" />
train_dataset[0] 返回啥？（答案：(img, label)，img 是 (1,28,28)）
len(train_dataset) 是多少？（答案：60000）
一个 batch 的 imgs.shape 是多少？（答案：(32, 1, 28, 28)）
B、C、H、W 分别是啥？（下面详细讲）
<img width="1109" height="526" alt="image" src="https://github.com/user-attachments/assets/4b7e22dd-d3cb-43ec-80b2-2967f2609a2b" />


## ***day3 cnn+训练循环*** 
<img width="1373" height="704" alt="image" src="https://github.com/user-attachments/assets/4ba4b5dc-7bd9-4e46-9999-8b07c8e29e61" />
<img width="1299" height="931" alt="image" src="https://github.com/user-attachments/assets/89cf85fa-30ae-4c38-b7d0-3b6dbdd29091" />

## ***day4 目标检测基础（yolo）*** 
概念
<img width="1134" height="578" alt="image" src="https://github.com/user-attachments/assets/011e369c-d322-48f3-a1f2-ce849d359837" />
<img width="1133" height="867" alt="image" src="https://github.com/user-attachments/assets/defff233-c9b7-457b-a0e1-684e989c7250" />

## ***day5 分割基础*** 
学习重点：Semantic Segmentation、mask、U-Net 思路。
动手任务：运行一个现成分割 demo，观察 mask。
今日产出：原图+mask。
今日验收：能解释“每个像素属于什么类别”。
<img width="1248" height="974" alt="image" src="https://github.com/user-attachments/assets/d28359ee-5ca4-40b2-9fc3-d3c556e0a3a6" />
<img width="1271" height="646" alt="image" src="https://github.com/user-attachments/assets/3db956b3-9b7a-4e86-92b8-039ba9ec6131" />

## ***day6 OCR + 地图小实验***
知识点
<img width="1175" height="434" alt="image" src="https://github.com/user-attachments/assets/7843c922-fa8f-44d5-97d6-01bbc9921fd9" />

遇到的困难
<img width="1184" height="732" alt="image" src="https://github.com/user-attachments/assets/e00d34ab-2700-48d4-966d-9b17599c676c" />

git技能
<img width="1177" height="580" alt="image" src="https://github.com/user-attachments/assets/da6b033b-cb8c-4cde-b856-da5a29c1c132" />

## ***day8 地图图像与遥感影像的差异***

今天学习了地图图像和遥感影像的区别。遥感影像记录地表外观，可以根据颜色、纹理和形状识别地物。地图经过筛选、概括和符号化，会突出需要表达的信息。

以运动场为例，遥感影像中可以看到跑道和草坪，地图中可能用简化图形和文字表示。换一种地图风格，颜色和符号可能变化，但运动场的位置及其与周边道路的关系通常仍有对应。

地图识别需要处理三个问题：同一对象有不同画法，相似符号可能含义不同，文字需要与对应对象关联。因此，需要结合图例、文字和空间关系理解地图。

## ***day9 地图五类特征与三张地图分析***

### 今日完成内容

学习地图字符、地图符号、地图场景、地图对象形态和地图类型。字符与符号是个体特征，场景与对象形态是局部特征，地图类型是全局特征。完成以下三张地图的人工分析。

#### 地图一：城市电子地图

[查看原图](pytorch_study/map_test.jpg)

- 地图字符：“北行涮串”对应餐饮店；“创新路”在道路注记和地铁站名中出现，需结合文字位置判断对应对象。
- 地图符号：地铁标志结合站名、线路编号和紫色线路表示地铁站；橙色餐饮图标结合店名表示餐饮地点。
- 地图场景：创新路站周边的城市街区，包含道路、建筑、餐饮店和交通设施。仅凭楼群与空地，尚不能确定是住宅小区。
- 地图对象形态：创新路在图中近似左右延伸，火石桥大街近似上下延伸，两者形成十字交叉，紫色线路沿火石桥大街方向延伸。
- 地图类型：初步判断为城市电子地图，带有三维建筑展示和公共交通信息。依据是同时展示道路、商铺、地铁站及立体建筑。截图缺少完整标题和图例。

#### 地图二：城市电子地图

[查看原图](pytorch_study/map_test2.jpg)

- 地图字符：“洗墨池路”和“运达路”对应道路；“30栋”和“22栋”是楼栋编号，需要与建筑轮廓关联。
- 地图符号：闪电图标结合“国家电网汽车充电站”文字表示充电设施；购物车图标结合超市名称表示购物地点。
- 地图场景：右上区域包含绿地、水域和内部通道，结合充电站注记中的“翊武公园”，初步判断为公园局部场景。
- 地图对象形态：洗墨池路与运达路形成十字交叉；水域呈不规则长条形，周边通道形成多个近似矩形区域。
- 地图类型：初步判断为城市电子地图，依据是展示道路名称、楼栋编号、商铺和公共设施，并区分绿地与水域。截图缺少完整标题和图例。

#### 地图三：中国省级行政区划图

[查看原图](images/day9_china_administrative_map.jpg)

- 地图字符：“四川”“湖北”“新疆”等标注行政区域名称；北京、天津、上海附近区域较小，文字与区域的对应需要仔细判断。
- 地图符号：不同颜色的面区分行政区域，边界线分隔相邻区域。颜色用于区分区域，不能据此判断人口或经济数值。
- 地图场景：东北局部由黑龙江、吉林、辽宁等相邻行政区域组成，体现区域尺度上的分布与组合。
- 地图对象形态：行政区域轮廓、海岸线与岛屿形态提供识别线索，例如狭长的内蒙古和独立岛状的海南，还需结合地名及邻接关系确认。
- 地图类型：省级行政区划图，依据是省级区域名称、边界和分区配色构成主要内容，整体突出行政区域的划分与分布。

### 遇到的问题

- 容易把道路两边的局部结构当成地图类型，也容易仅凭建筑与空地就判断为小区。
- 不清楚“根据任务选择特征和方法”的含义，以及检测、分割、OCR和对象关联的分工。
- 道路或楼栋即使没有独立图标，也可能通过线条、轮廓和文字表达，需要判断文字对应哪个对象。

### 解决情况

- 场景关注局部区域的要素组合，形态关注排列、轮廓与交叉结构，类型关注整张地图的表达主题和用途。判断需要记录图上证据及不确定性。
- 任务决定所需方法。例如读取省名可用OCR，提取省域可利用颜色、边界进行区域提取或分割，判断相邻需要分析区域边界关系。
- 餐饮店提取可采用“检测餐饮图标 → OCR读取店名 → 结合位置与语义进行关联”的流程。主要风险包括图标漏检、文字识别错误和图文关联错误。
- 检测模型通过带框和类别的样本学习图标外观。通用YOLO权重是否适用，需要检查支持类别并实测，地图风格变化也可能影响识别。
- 本日为人工观察与方法分析，尚未训练餐饮图标模型或运行上述组合流程。

### 明日计划

进入Day10，阅读《Recognition and Semantic Information Extraction for Map Based on Deep Learning》的摘要、引言、结论和主要图，填写论文阅读卡前八项，将论文的方法分类与已有视觉基础联系起来。
