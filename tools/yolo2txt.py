import os
import random

# 设置路径
dataset_path = '/path/to/my_dataset'
image_dir = os.path.join(dataset_path, 'images')
output_dir = dataset_path

# 获取所有图像文件
all_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
             if f.endswith(('.jpg', '.jpeg', '.png'))]

# 随机打乱
random.shuffle(all_images)

# 按8:2比例分割
split_idx = int(len(all_images) * 0.8)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# 写入文件
with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
    for path in train_images:
        f.write(path + '\n')

with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
    for path in val_images:
        f.write(path + '\n')

print(f"生成完成！训练集: {len(train_images)}，验证集: {len(val_images)}")