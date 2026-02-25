import os
import json
import argparse
from PIL import Image
from collections import defaultdict

import os
import json
import cv2
from collections import defaultdict
from pathlib import Path
from datetime import datetime

current_date = datetime.now().strftime("%Y-%m-%d")

def yolo_to_coco_with_emptyfile(data_root, output_json_path):
    """
    将 YOLO 格式数据集（包含空标签文件）转换为 COCO 格式
    
    Args:
        data_root (str): 数据集根目录，包含 'train', 'val', 'test' 子目录
        output_json_path (str): 输出的 COCO JSON 文件路径
    """
    splits = ['train', 'val', 'test']  # 假设你有这些分割
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 初始化 COCO JSON 结构:cite[3]
    coco_data = {
        "info": {
            "description": "Emphysema Dataset",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "",
            "date_created": "2025-09-07"
        },
        "licenses": [{
            "id": 1,
            "name": "License",
            "url": ""
        }],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "emphysema", "supercategory": "none"},
            {"id": 1, "name": "normal", "supercategory": "none"}
        ]
    }
    
    annotation_id = 1  # 标注 ID 从 1 开始
    image_id = 1       # 图像 ID 从 1 开始

    for split in splits:
        split_image_dir = Path(data_root) / split / 'images'
        split_label_dir = Path(data_root) / split / 'labels'
        
        # 获取所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(split_image_dir.glob(f'*{ext}')))
            image_files.extend(list(split_image_dir.glob(f'*{ext.upper()}')))
        
        for img_path in image_files:
            # 读取图像获取宽高
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read image {img_path}, skipping.")
                continue
            height, width = img.shape[:2]
            
            # 添加图像信息到 COCO
            image_info = {
                "id": image_id,
                "file_name": str(img_path.relative_to(data_root).as_posix()),  # 相对路径
                "width": width,
                "height": height,
                "license": 1,
                "date_captured": "2025-09-07"
            }
            coco_data["images"].append(image_info)
            
            # 处理对应的标签文件
            label_path = split_label_dir / f"{img_path.stem}.txt"
            if label_path.exists() and label_path.stat().st_size > 0:
                # 标签文件存在且非空，读取并转换标注
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"Warning: Invalid line format in {label_path}: {line}")
                        continue
                    
                    try:
                        class_id, x_center, y_center, w, h = map(float, parts)
                    except ValueError:
                        print(f"Warning: Invalid number in {label_path}: {line}")
                        continue
                    
                    # 将 YOLO 相对坐标转换为 COCO 绝对坐标:cite[3]
                    x_center_abs = x_center * width
                    y_center_abs = y_center * height
                    w_abs = w * width
                    h_abs = h * height
                    
                    x_min = max(0, x_center_abs - w_abs / 2)
                    y_min = max(0, y_center_abs - h_abs / 2)
                    
                    # 确保边界框不超出图像范围
                    x_min = max(0, min(x_min, width - 1))
                    y_min = max(0, min(y_min, height - 1))
                    w_abs = max(1, min(w_abs, width - x_min))
                    h_abs = max(1, min(h_abs, height - y_min))
                    
                    area = w_abs * h_abs
                    
                    # 添加标注信息到 COCO
                    annotation_info = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(class_id),
                        "bbox": [x_min, y_min, w_abs, h_abs],
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": []
                    }
                    coco_data["annotations"].append(annotation_info)
                    annotation_id += 1
            # 如果标签文件不存在或为空（空标签文件），则不在 annotations 中添加任何内容
            image_id += 1
    
    # 保存 COCO JSON 文件
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    print(f"Conversion complete. COCO JSON saved to: {output_json_path}")

# def yolo_to_coco(yolo_dir,output_file,class_names_file=None):
#     """
#     将YOLO格式标注转换为COCO检测数据格式
    
#     参数:
#         yolo_dir: YOLO格式数据目录，包含images和labels文件夹
#         output_file: 输出的COCO格式JSON文件路径
#         class_names_file: YOLO类别名称文件路径(可选)
#     """

#     # 确定图像和标注目录
#     images_dir = os.path.join(yolo_dir,'images')
#     labels_dir = os.path.join(yolo_dir,'labels')

#     # 确定目录存在
#     if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
#         raise ValueError(f" {yolo_dir} 目录必须包含'images'和'labels'文件夹")

#     # 获取所有图像文件
#     image_extensions = ['.jpg','.jpeg','.png','.bmp']
#     image_files = []
#     for ext in image_extensions:
#         image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])
        
#     # 读取类别名称
#     if class_names_file and os.path.exists(class_names_file):
#         with open(class_names_file,'r') as f:
#             classes = [line.strip() for line in f.readlines() if line.strip()]
#     else:
#         # 如果没有提供类别文件，尝试从标注中推断
#         classes_ids = set()
#         for img_file in image_files:
#             label_file = os.path.splitext(img_file)[0] + '.txt'
#             label_path = os.path.join(labels_dir, label_file)

#             if os.path.exists(label_path):
#                 with open(label_path, 'r') as f:
#                     for line in f:
#                         parts = line.strip().split()
#                         if parts:
#                             class_id = int(parts[0])
#                             classes_ids.add(str(class_id))
#         # 将集合转换为排序列表
#         classes_ids = sorted(list(classes_ids))
#         # 转换为类别名称 (如果没有提供名称文件，使用数字作为名称)
#         classes = [f"class_{cls}" for cls in classes_ids]

#     print(classes)

#     # 创建coco数据结构
#     coco_data = {
#         "images": [],
#         "annotations": [],
#         "categories": [],
#         "info": {
#             "description": "COCO dataset converted from YOLO format",
#             "version": "1.0",
#             "year": 2023,
#             "contributor": "YOLO to COCO converter"
#         },
#         "licenses": [{"id": 1, "name": "Unknown License"}]
#     }

#     # 添加类别信息
#     for i, class_name in enumerate(classes):
#         coco_data["categories"].append({
#             "id": i + 1,  # COCO类别ID从1开始
#             "name": class_name,
#             "supercategory": "none"
#         })

#     # 创建类别ID映射 （Yolo 类别ID到COCO类别ID）
#     category_mapping = {}
#     for i in range(len(classes)):
#         category_mapping[i] = i + 1 # YOLO ID --> COCO ID
    

#     # 处理每个图像
#     annotation_id = 1
#     for image_id,img_file in enumerate(image_files,1):
#         img_path = os.path.join(images_dir,img_file)

#         # 获取图像尺寸
#         try:
#             with Image.open(img_path) as img:
#                 width,height = img.size
#         except Exception as e:
#             print(f"无法读取图像 {img_file} : {e}")
#             continue

#         # 添加图像信息到COCO
#         coco_data["images"].append({
#             "id": image_id,
#             "width": width,
#             "height": height,
#             "file_name": img_file,
#             "license": 1,
#             "date_captured": "2025-09-07"
#         })

#         # 处理对应的标注文件
#         label_file = os.path.splitext(img_file)[0] + '.txt'
#         label_path = os.path.join(labels_dir,label_file)

#         if not os.path.exists(label_path):
#             continue

#         with open(label_path,'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) < 5:
#                     continue

#                 try:
#                     # 解析YOLO格式
#                     class_id = int(parts[0])
#                     x_center = float(parts[1])
#                     y_center = float(parts[2])
#                     bbox_width = float(parts[3])
#                     bbox_height = float(parts[4])

#                     # 转换为COCO格式（绝对坐标）
#                     x = (x_center - bbox_width / 2 ) * width
#                     y = (y_center - bbox_height / 2 ) * height
#                     w = bbox_width * width
#                     h = bbox_height * height

#                     # 确保坐标在图像范围内
#                     x = max(0, min(x, width))
#                     y = max(0, min(y, height))
#                     w = max(0, min(w, width - x))
#                     h = max(0, min(h, height - y))

#                     # 添加标注信息到COCO
#                     coco_data["annotations"].append({
#                         "id": annotation_id,
#                         "image_id": image_id,
#                         "category_id": category_mapping.get(class_id, 0),
#                         "gt_bbox": [x, y, w, h],
#                         "area": w * h,
#                         "is_crowd": 0,
#                         "segmentation": []
#                     })

#                     annotation_id += 1
#                 except ValueError as e:
#                     print(f"解析标注错误 {label_path} : {e}")
#                     continue


#     # 保存COCO格式json文件
#     with open(output_file, 'w') as f:
#         json.dump(coco_data, f, indent=2)
    
#     print(f"转换完成! 共处理 {len(coco_data['images'])} 张图像, {len(coco_data['annotations'])} 个标注")
#     print(f"结果已保存到: {output_file}")

def yolo_to_coco(yolo_dir, output_file, class_names_file=None):
    """
    将YOLO格式标注转换为COCO检测数据格式
    
    参数:
        yolo_dir: YOLO格式数据目录，包含images和labels文件夹
        output_file: 输出的COCO格式JSON文件路径
        class_names_file: YOLO类别名称文件路径(可选)
    """

    # 确定图像和标注目录
    images_dir = os.path.join(yolo_dir, 'images')
    labels_dir = os.path.join(yolo_dir, 'labels')

    # 确定目录存在
    if not os.path.exists(images_dir):
        raise ValueError(f"{images_dir} 目录不存在")
    if not os.path.exists(labels_dir):
        print(f"警告: {labels_dir} 目录不存在，将创建空目录")
        os.makedirs(labels_dir, exist_ok=True)

    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        raise ValueError(f"{images_dir} 目录中没有找到图像文件")
        
    print(f"找到 {len(image_files)} 个图像文件")

    # 读取类别名称
    if class_names_file and os.path.exists(class_names_file):
        with open(class_names_file, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
    else:
        # 如果没有提供类别文件，尝试从标注中推断
        classes_ids = set()
        for img_file in image_files:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)

            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                class_id = int(parts[0])
                                classes_ids.add(class_id)
                            except ValueError:
                                print(f"警告: 无法解析类别ID: {parts[0]}")
        # 将集合转换为排序列表
        classes_ids = sorted(list(classes_ids))
        # 转换为类别名称 (如果没有提供名称文件，使用数字作为名称)
        classes = [f"class_{cls}" for cls in classes_ids]
        
        # 如果没有找到任何类别，添加默认类别
        if not classes:
            classes = ["emphysema", "normal"]
            print("警告: 未找到任何类别，使用默认类别: ['emphysema', 'normal']")

    print(f"检测到的类别: {classes}")

    # 创建coco数据结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [],
        "info": {
            "description": "COCO dataset converted from YOLO format",
            "version": "1.0",
            "year": 2023,
            "contributor": "YOLO to COCO converter"
        },
        "licenses": [{"id": 1, "name": "Unknown License"}]
    }

    # 添加类别信息
    for i, class_name in enumerate(classes):
        coco_data["categories"].append({
            "id": i,  # COCO类别ID从0开始
            "name": class_name,
            "supercategory": "none"
        })

    # 处理每个图像
    annotation_id = 0
    image_id = 0

    processed_images = 0
    processed_annotations = 0
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)

        # 获取图像尺寸
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"无法读取图像 {img_file} : {e}")
            continue

        # 添加图像信息到COCO
        coco_data["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": img_file,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_created":str(current_date) 
        })
        processed_images += 1

        # 处理对应的标注文件
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        # 检查标签文件是否存在且不为空
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                        
                    if len(parts) < 5:
                        print(f"警告: {label_path} 第 {line_num} 行格式错误: {line}")
                        continue

                    try:
                        # 解析YOLO格式
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        bbox_width = float(parts[3])
                        bbox_height = float(parts[4])

                        # 将YOLO格式转换为COCO格式（绝对坐标）
                        x = (x_center - bbox_width / 2) * width
                        y = (y_center - bbox_height / 2) * height
                        w = bbox_width * width
                        h = bbox_height * height

                        # 确保坐标在图像范围内
                        x = max(0, min(x, width))
                        y = max(0, min(y, height))
                        w = max(0, min(w, width - x))
                        h = max(0, min(h, height - y))

                        # 跳过无效的边界框
                        if w <= 0 or h <= 0:
                            print(f"警告: {label_path} 第 {line_num} 行边界框无效")
                            continue

                        # 添加标注信息到COCO
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,  # 直接使用YOLO类别ID
                            "bbox": [x, y, w, h],  # 使用标准COCO字段名"bbox"
                            "area": w * h,
                            "iscrowd": 0,
                            "segmentation": []
                        })

                        annotation_id += 1
                        processed_annotations += 1
                    except ValueError as e:
                        print(f"解析标注错误 {label_path} 第 {line_num} 行: {e}")
                        continue
        else:
            # 标签文件不存在或为空，表示这是一个正常/无目标的图像
            # 在COCO格式中，不需要为这类图像添加任何标注
            # 对于正常图像（无标注），添加一个空的标注条目
            # coco_data["annotations"].append({
            #     "id": annotation_id,
            #     "image_id": image_id,
            #     "category_id": 1,  # 假设正常类别ID为1
            #     "bbox": [],  # 空bbox
            #     "area": 0,
            #     "segmentation": [],
            #     "iscrowd": 0
            # })
            annotation_id += 1
        image_id += 1
            

    # 保存COCO格式json文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成! 共处理 {processed_images} 张图像, {processed_annotations} 个标注")
    print(f"结果已保存到: {output_file}")
    
    # 返回统计信息
    return {
        "images": processed_images,
        "annotations": processed_annotations,
        "categories": len(classes)
    }

def parse():
    parser = argparse.ArgumentParser(description="将YOLO格式转换为COCO检测格式")
    parser.add_argument(
        "--yolo_dir", 
        default="data/exp1_test/val",
        # required=True, 
        help="YOLO格式数据目录")
    parser.add_argument(
        "--output", 
        default="data/exp1_test/val/val.json",
        # required=True, 
        help="输出JSON文件路径")
    parser.add_argument(
        "--classes",
        default="data/exp1_test/classes.txt",
        help="YOLO类别名称文件路径(可选)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse()

    yolo_to_coco(args.yolo_dir,args.output,args.classes)


