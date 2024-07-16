import os
import shutil

def rename_and_move_images(source_directory, destination_directory):
    # 遍历源文件夹中的所有子文件夹
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            # 检查文件是否为图像文件（这里只考虑了常见的图像格式）
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                # 获取子文件夹的名字
                subfolder_name = os.path.basename(root)
                
                # 组合新的文件名
                new_filename = f"{subfolder_name}_{file}"
                
                # 构建目标路径
                destination_path = os.path.join(destination_directory, new_filename)
                
                # 移动文件并重命名
                shutil.move(file_path, destination_path)

# 设置源文件夹和目标文件夹的路径

# source_directory = "C:/Users/coco/Desktop/keypoint/patch"
# destination_directory = "C:/Users/coco/Desktop/keypoint/patches"

source_directory = "patch"
destination_directory = "patches"

# 调用函数重命名和移动图像文件
rename_and_move_images(source_directory, destination_directory)