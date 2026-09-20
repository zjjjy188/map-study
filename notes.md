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

