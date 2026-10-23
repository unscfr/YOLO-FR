import os
import csv
from ultralytics import YOLO

# ===============================
# 1. 模型与路径配置
# ===============================
weights_path = r"/data/FR/FRYOLO/train/runs/data/data-yolo11m-1-500/weights/best.pt"  # 训练好的YOLO模型权重
images_dir = r"/home/nxnu/Documents/val-images"  # 要预测的图片文件夹
output_csv = r"./pred_results.csv"  # 输出CSV文件路径

# 加载模型
model = YOLO(weights_path)

# ===============================
# 2. 预测与结果处理
# ===============================
# 写入CSV的表头
header = ["image_id", "class_id", "x_center", "y_center", "width", "height"]

with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    # 遍历图片文件夹
    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif')):
            continue

        img_path = os.path.join(images_dir, img_name)
        results = model.predict(img_path, imgsz=640, conf=0.25, verbose=False)[0]

        # 获取检测结果
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            continue

        # 提取置信度并按降序排序
        confs = boxes.conf.cpu().numpy()
        sorted_idx = confs.argsort()[::-1]
        top_idx = sorted_idx[:4]  # 取置信度最高的前4个

        for i in top_idx:
            box = boxes[i]
            cls_id = int(box.cls)
            x_center, y_center, w, h = box.xywhn[0].tolist()  # 归一化坐标

            writer.writerow([img_name, cls_id, x_center, y_center, w, h])

print(f"预测完成，结果已保存至：{output_csv}")
