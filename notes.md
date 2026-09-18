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
