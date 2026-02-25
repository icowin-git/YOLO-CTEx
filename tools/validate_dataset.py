import os
import cv2
import numpy as np
from datetime import datetime



def validate_yolo_dataset(image_dir, label_dir, log_file=None):
    """
    验证YOLO数据集完整性并统计各种情况，将错误写入日志文件
    
    参数:
        image_dir: 图像文件夹路径
        label_dir: 标签文件夹路径
        log_file: 日志文件路径（可选）
    """
    # 如果没有指定日志文件，使用默认名称
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"yolo_dataset_validation_{timestamp}.log"
    
    # 创建日志文件夹（如果不存在）
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"创建日志文件夹: {log_dir}")
    
    # 打开日志文件
    with open(log_file, 'w', encoding='utf-8') as log:
        log.write(f"YOLO数据集验证日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"图像目录: {image_dir}\n")
        log.write(f"标签目录: {label_dir}\n")
        log.write("=" * 50 + "\n\n")
    
    # 获取所有图像和标签文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    print(f"发现 {len(image_files)} 张图像")
    print(f"发现 {len(label_files)} 个标签文件")
    
    # 初始化统计计数器
    stats = {
        'total_images': len(image_files),
        'total_labels': len(label_files),
        'images_with_labels': 0,      # 有对应标签文件的图像
        'images_without_labels': 0,   # 没有对应标签文件的图像
        'empty_labels': 0,            # 标签文件为空
        'invalid_format': 0,          # 标签格式错误
        'out_of_bounds': 0,           # 坐标越界
        'total_bboxes': 0,            # 总边界框数量
        'valid_images': 0             # 完全有效的图像
    }
    
    # 创建详细错误记录
    errors = {
        'missing_labels': [],
        'empty_labels': [],
        'invalid_format': [],
        'out_of_bounds': []
    }
    
    # 检查每个图像文件
    for img_file in image_files:
        # 检查标签文件是否存在
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        if not os.path.exists(label_path):
            stats['images_without_labels'] += 1
            error_msg = f"缺失标签文件: {label_file} (对应图像: {img_file})"
            errors['missing_labels'].append(error_msg)
            
            # 写入日志
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"❌ {error_msg}\n")
            continue
            
        stats['images_with_labels'] += 1
        
        # 读取标签内容
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            stats['empty_labels'] += 1
            error_msg = f"空标签文件: {label_file} (对应图像: {img_file})"
            errors['empty_labels'].append(error_msg)
            
            # 写入日志
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"❌ {error_msg}\n")
            continue
            
        # 验证标签内容
        is_valid = True
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                stats['invalid_format'] += 1
                error_msg = f"格式错误 {label_file}:{i+1} - '{line.strip()}' (对应图像: {img_file})"
                errors['invalid_format'].append(error_msg)
                
                # 写入日志
                with open(log_file, 'a', encoding='utf-8') as log:
                    log.write(f"❌ {error_msg}\n")
                is_valid = False
                continue
                
            # 检查坐标是否在0-1之间
            try:
                coords = list(map(float, parts[1:]))
                if any(coord < 0 or coord > 1 for coord in coords):
                    stats['out_of_bounds'] += 1
                    error_msg = f"坐标越界 {label_file}:{i+1} - {coords} (对应图像: {img_file})"
                    errors['out_of_bounds'].append(error_msg)
                    
                    # 写入日志
                    with open(log_file, 'a', encoding='utf-8') as log:
                        log.write(f"❌ {error_msg}\n")
                    is_valid = False
            except ValueError:
                stats['invalid_format'] += 1
                error_msg = f"格式错误(数值转换失败) {label_file}:{i+1} - '{line.strip()}' (对应图像: {img_file})"
                errors['invalid_format'].append(error_msg)
                
                # 写入日志
                with open(log_file, 'a', encoding='utf-8') as log:
                    log.write(f"❌ {error_msg}\n")
                is_valid = False
                
            stats['total_bboxes'] += 1
        
        if is_valid:
            stats['valid_images'] += 1
            # 写入日志 - 有效文件
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"✅ {img_file} 验证通过\n")
    
    # 打印统计结果
    print("\n=== 数据集统计结果 ===")
    print(f"总图像数量: {stats['total_images']}")
    print(f"总标签文件数量: {stats['total_labels']}")
    print(f"有标签文件的图像: {stats['images_with_labels']} ({stats['images_with_labels']/stats['total_images']*100:.1f}%)")
    print(f"无标签文件的图像: {stats['images_without_labels']} ({stats['images_without_labels']/stats['total_images']*100:.1f}%)")
    print(f"空标签文件: {stats['empty_labels']}")
    print(f"格式错误的标签: {stats['invalid_format']}")
    print(f"坐标越界的标签: {stats['out_of_bounds']}")
    print(f"总边界框数量: {stats['total_bboxes']}")
    print(f"完全有效的图像: {stats['valid_images']} ({stats['valid_images']/stats['total_images']*100:.1f}%)")
    
    # 将统计结果写入日志
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write("\n" + "=" * 50 + "\n")
        log.write("数据集统计结果:\n")
        log.write("=" * 50 + "\n")
        log.write(f"总图像数量: {stats['total_images']}\n")
        log.write(f"总标签文件数量: {stats['total_labels']}\n")
        log.write(f"有标签文件的图像: {stats['images_with_labels']} ({stats['images_with_labels']/stats['total_images']*100:.1f}%)\n")
        log.write(f"无标签文件的图像: {stats['images_without_labels']} ({stats['images_without_labels']/stats['total_images']*100:.1f}%)\n")
        log.write(f"空标签文件: {stats['empty_labels']}\n")
        log.write(f"格式错误的标签: {stats['invalid_format']}\n")
        log.write(f"坐标越界的标签: {stats['out_of_bounds']}\n")
        log.write(f"总边界框数量: {stats['total_bboxes']}\n")
        log.write(f"完全有效的图像: {stats['valid_images']} ({stats['valid_images']/stats['total_images']*100:.1f}%)\n")
    
    # 打印错误摘要
    print("\n=== 错误摘要 ===")
    for error_type, items in errors.items():
        if items:
            print(f"{error_type}: {len(items)} 个错误")
            # 写入日志
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"\n{error_type}: {len(items)} 个错误\n")
            
            if len(items) <= 5:  # 只显示前5个错误
                for item in items[:5]:
                    print(f"  - {item}")
                    with open(log_file, 'a', encoding='utf-8') as log:
                        log.write(f"  - {item}\n")
            else:
                print(f"  - 前5个错误:")
                with open(log_file, 'a', encoding='utf-8') as log:
                    log.write(f"  - 前5个错误:\n")
                for item in items[:5]:
                    print(f"    - {item}")
                    with open(log_file, 'a', encoding='utf-8') as log:
                        log.write(f"    - {item}\n")
                print(f"  - ... 还有 {len(items)-5} 个错误未显示")
                with open(log_file, 'a', encoding='utf-8') as log:
                    log.write(f"  - ... 还有 {len(items)-5} 个错误未显示\n")
        else:
            print(f"{error_type}: 无错误")
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"{error_type}: 无错误\n")
    
    # 计算数据集质量评分 (0-100)
    quality_score = (stats['valid_images'] / stats['total_images']) * 100
    print(f"\n数据集质量评分: {quality_score:.1f}/100")
    
    # 根据评分给出建议
    if quality_score > 90:
        assessment = "数据集质量优秀"
    elif quality_score > 70:
        assessment = "数据集质量一般，建议修复部分问题"
    else:
        assessment = "数据集质量较差，需要大量修复工作"
    
    print(assessment)
    
    # 将评估结果写入日志
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f"\n数据集质量评分: {quality_score:.1f}/100\n")
        log.write(f"评估: {assessment}\n")
    
    # 生成修复建议
    generate_fix_suggestions(stats, errors, log_file)
    
    print(f"\n详细错误日志已保存至: {os.path.abspath(log_file)}")
    
    return stats, errors, log_file

def generate_fix_suggestions(stats, errors, log_file):
    """生成修复建议并写入日志"""
    suggestions = []
    
    if stats['images_without_labels'] > 0:
        suggestions.append(f"1. 有 {stats['images_without_labels']} 张图像缺少标签文件，需要补充标注或删除这些图像")
    
    if stats['empty_labels'] > 0:
        suggestions.append(f"2. 有 {stats['empty_labels']} 个标签文件为空，需要补充标注或删除这些文件")
    
    if stats['invalid_format'] > 0:
        suggestions.append(f"3. 有 {stats['invalid_format']} 个标签格式错误，需要按照YOLO格式修正（类别ID 中心X 中心Y 宽度 高度）")
    
    if stats['out_of_bounds'] > 0:
        suggestions.append(f"4. 有 {stats['out_of_bounds']} 个标签坐标越界（不在0-1范围内），需要修正坐标值")
    
    if not suggestions:
        suggestions.append("数据集无需修复，可以直接使用")
    
    # 写入建议到日志
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write("\n" + "=" * 50 + "\n")
        log.write("修复建议:\n")
        log.write("=" * 50 + "\n")
        for suggestion in suggestions:
            log.write(f"{suggestion}\n")
    
    # 打印建议
    print("\n=== 修复建议 ===")
    for suggestion in suggestions:
        print(suggestion)

# 使用示例
if __name__ == "__main__":
    
    # 使用示例
    image_dir = '/data-share/sgri_zhangqiang/projects/Emphysema.Detection/data/exp1_test/val/images'
    label_dir = '/data-share/sgri_zhangqiang/projects/Emphysema.Detection/data/exp1_test/val/labels'
    log_file = './logs/yolo_dataset_validation.log'
    stats, errors, log_file = validate_yolo_dataset(image_dir, label_dir,log_file)

