import cv2
import os
from ultralytics import YOLO

# 加载模型
#model = YOLO(r"C:\Users\FR\Desktop\范瑞论文\c4c结果\VIS-FL-EIoU-yolo11n-4c4c-carafee-SSS-ConcatPro-800-0.37843-0.22626\weights\best.pt")

# class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van',
#                'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']


# 设置图像目录和输出目录
image_dir = r"/data/FR/datasets/AI-TOD/val/images"
output_dir = r"/data/FR/datasets/AI-TOD/val/groundtruth"

# 如果输出目录不存在，则创建
os.makedirs(output_dir, exist_ok=True)

# 获取图像目录下所有图片的路径
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
#image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

# 遍历每一张图片
for image_path in image_paths:
    # 读取图像
    image = cv2.imread(image_path)

    # 获取图像的标注文件路径
    label_path = image_path.replace('images', 'labels').replace('.png', '.txt')
    #label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')

    # 检查标注文件是否存在
    if not os.path.exists(label_path):
        print(f"Label file {label_path} not found. Skipping...")
        continue

    # 读取标注文件
    with open(label_path, 'r') as f:
        labels = f.readlines()

    # 将原始标注绘制到图像上
    for label in labels:
        # 标注格式: class x_center y_center width height (归一化)
        class_id, x_center, y_center, width, height = map(float, label.strip().split())

        # 获取图像的尺寸
        h, w, _ = image.shape

        # 将归一化的坐标转换为实际像素坐标
        x_center = int(x_center * w)
        y_center = int(y_center * h)
        width = int(width * w)
        height = int(height * h)

        # 计算左上角和右下角坐标
        x1 = x_center - width // 2
        y1 = y_center - height // 2
        x2 = x_center + width // 2
        y2 = y_center + height // 2

        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框表示原始标注

        # 显示类别名称（类别ID映射到类别名称）
        #label_name = class_names[int(class_id)]  # 获取类别名称
        #cv2.putText(image, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 在图像上执行对象检测
    #results = model(image_path)

    # 显示检测结果
    #results[0].show()

    # 保存带有原标注和预测结果的图像
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)
    print(f"Processed and saved: {output_image_path}")