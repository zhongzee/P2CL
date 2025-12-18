import os

def list_subdirectories(path):
    """
    遍历指定路径下的所有子目录并返回它们的列表
    """
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def main():
    path = '/root/autodl-tmp/office31/amazon'
    subdirs = list_subdirectories(path)

    # 将子目录名称写入txt文件，并用逗号分隔
    with open('office31_txt_labels.txt', 'w') as f:
        f.write(', '.join(['"' + subdir + '"' for subdir in subdirs]))

if __name__ == "__main__":
    main()
