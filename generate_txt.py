import os

# 定义数据集的路径
# base_dataset_path = '/root/autodl-tmp/office31'
base_dataset_path = '/root/autodl-tmp/image_CLEF'

# 获取所有图片文件的扩展名
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']


def collect_image_paths(dir_path):
    """递归地收集目录中的所有图片路径"""
    image_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths


# 遍历base_dataset_path下的每个子目录
for subdir in os.listdir(base_dataset_path):
    subdir_path = os.path.join(base_dataset_path, subdir)

    # 检查是否为目录
    if os.path.isdir(subdir_path):
        # 收集子目录中的所有图片文件路径
        image_paths = collect_image_paths(subdir_path)

        # 为子目录生成txt文件
        txt_file_path = os.path.join(base_dataset_path, f"{subdir}_reorgnized.txt")
        with open(txt_file_path, 'w') as txt_file:
            for image_path in image_paths:
                txt_file.write(image_path + '\n')

        print(f"{subdir}_reorgnized.txt has been generated with {len(image_paths)} image paths.")

print("All txt files have been generated.")
