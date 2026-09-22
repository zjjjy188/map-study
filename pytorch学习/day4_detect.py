from ultralytics import YOLO
import cv2
print("="*50)
print("Day4: YOLO 目标检测实战")
print("="*50)
# 1. 加载模型
# "yolov8n.pt" 是 YOLO 最小的模型（n=nano，纳米级）。第一次运行会自动下载这个权重文件。
# 这里就是“推理”的入口：我们直接使用别人训练好的模型，不自己训练。
# **权重文件（.pt）**：就是模型训练完学到的 “记忆”，里面存着识别物体的参数。
# **推理 (inference)**：用已经训练好的模型，去识别新图片，**不训练、不更新模型参数**。
model = YOLO("yolov8l.pt")
# 2. 读取图片
# 我们要检测的图片路径
image_path = "test2.jpg"
# 3. 进行预测（推理）
# 把图片扔给模型，模型会自动画好框
print(f"正在检测：{image_path}...")
results = model(image_path)
# 4. 提取检测结果并打印
for result in results:
    boxes = result.boxes  # 获取所有检测到的边界框（Bounding Box）
    print(f"\n共检测到{len(boxes)}个物体")
    for i, box in enumerate(boxes): # enumerate 可以同时拿到序号 i 和 每一个框 box。
        # 提取坐标
        x1, y1, x2, y2 = box.xyxy[0].tolist()#`box.xyxy`：张量，存储坐标；`.tolist()` 把张量转为普通 Python 数字列表。
        # 提取置信度（Confidence Score）
        conf=box.conf[0].item()#置信度 conf：模型对这个检测结果的自信程度，0~1 之间。
        # 提取类别（Class）
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        print(f"  物体 {i+1}: {cls_name}")
        print(f"    坐标: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        print(f"    置信度: {conf:.2f}")
# 5. 保存画好框的结果图片（今日产出）
# result.save() 会自动把带有框和标签的图片保存下来
result.save(filename="detection_result.jpg")
print("\n✅ 检测完成！结果已保存为 detection_result.jpg")



