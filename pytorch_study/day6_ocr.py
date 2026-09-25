import easyocr
import cv2
import json
from PIL import Image
import numpy as np

print("=" * 50)
print("Day6: OCR 文字识别实战")
print("=" * 50)

# 1. 初始化 OCR 阅读器
# ['ch_sim', 'en'] 表示同时支持简体中文和英文。
# 第一次运行会下载模型文件（大约几十 MB）。
reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)

# 2. 读取图片
image_path = "map_test.jpg"
print(f"正在读取图片：{image_path}")
pil_image = Image.open(image_path).convert("RGB")
# PIL -> numpy
image = np.array(pil_image)
# RGB -> BGR
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# 3. 进行 OCR 识别
# *这一句同时完成了“文本检测”和“文本识别”。
print("正在进行OCR检测与识别...")
results = reader.readtext(image_path) # *
print(f"\n共检测到 {len(results)} 个文本框：")

# 用于保存 JSON 的列表
json_output = []

# 4. 遍历结果，画框并记录文字
for i,res in enumerate(results):
    # 提取坐标点、文字、置信度
    box = res[0]  # 这是一个包含 4 个点坐标的列表,一个长方形，给了四个角的坐标。
    text = res[1]  # 识别出来的文字
    confidence = res[2]  # 置信度（0~1）

    print(f"  文字 {i + 1}: '{text}'  (置信度: {confidence:.2f})")
    if confidence < 0.3:
        print("    -> 置信度太低，跳过")
        continue
    # 保存到 JSON
    json_output.append({
        "text": text,
        "confidence": round(confidence, 2),
        "box": [[int(x), int(y)] for x, y in box]  # 把坐标转成整数
    })

    # 用 OpenCV 画框
    # 把 4 个点连成一个矩形轮廓
    top_left = (int(box[0][0]), int(box[0][1]))
    bottom_right = (int(box[2][0]), int(box[2][1]))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # 绿色框，线宽 2
    # 在框上面写文字
    cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # 红色字

# 5. 保存结果
# 保存画好框的图片（今日产出 1）
cv2.imwrite("ocr_result.jpg", image)
print("\n✅ 检测结果图已保存为 ocr_result.jpg")

# 保存 JSON 文件（今日产出 2）
with open("ocr_result.json", "w", encoding="utf-8") as f:
    json.dump(json_output, f, ensure_ascii=False, indent=2)
print("✅ 文字坐标 JSON 已保存为 ocr_result.json")