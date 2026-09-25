from ultralytics import YOLO

from day4_detect import image_path

print("=" * 50)
print("Day5: YOLO 语义分割实战")
print("=" * 50)
# 1. 加载模型
# "yolov8n-seg.pt" 是 YOLOv8 的“分割”版本（seg = segmentation）。
# 注意这里是 -seg，不是普通的 yolov8n.pt。
model=YOLO("yolov8l-seg.pt")
# 2. 进行预测（推理）
image_path="test2.jpg"
print(f"正在检测并分割: {image_path} ...")
results = model(image_path)
# 3. 提取结果并打印信息
for result in results:
    # result.masks 就是分割出来的掩码信息
    if result.masks is not None:#如果 masks 不为空，也就是**成功识别到物体并生成掩码**，才执行 if 里面代码。
        # 掩码数据本身是一个矩阵，形状为 (物体数量, 高, 宽)
        masks=result.masks.data  #返回掩码张量（tensor，pytorch 里面的多维数组）维度：`(物体数量，图片高度，图片宽度)`
        print(f"\n共分割出{len(masks)}个物体。")
        # 获取每个物体的类别
        for i,mask in enumerate(masks):
            cls_id = int(result.boxes.cls[i].item())
            cls_name = model.names[cls_id]
            # 计算这个掩码覆盖了多少像素
            pixel_count = mask.sum().item()
            print(f"  物体 {i+1}: {cls_name}，掩码包含 {int(pixel_count)} 个像素")
    else:
        print("未检测到任何物体。")
# 4. 保存结果（今日产出：原图 + Mask 叠加图）
# save() 会自动把原图、掩码、框叠加在一起保存
result.save(filename="seg_result.jpg")
print("\n✅ 分割完成！结果已保存为 seg_result.jpg")


