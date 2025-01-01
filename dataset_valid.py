import os
import json

# Lasot
def list_subfolders_lasot(root_dir):
    subfolders_dict = {}
    class_sample = os.listdir(root_dir)
    for class_name in class_sample:
        subfolder_path = os.path.join(root_dir, class_name)
        subfolders_dict[class_name] = sorted(os.listdir(subfolder_path), key=lambda x: int(x.split('-')[1]))
    return subfolders_dict

# 替换为你的大文件夹路径
root_directory = '/data/guohua/BeiJing/data/lasot/'
result = list_subfolders_lasot(root_directory)
# print(result)

# 将结果字典写入JSON文件
json_filename = './lasot.json'  # 你可以根据需要修改文件名
with open(json_filename, 'w') as json_file:
    json.dump(result, json_file, indent=4)  # indent=4 用于格式化输出，使JSON文件易于阅读


# NFS
def list_subfolders_nfs(root_dir):
    subfolders_dict = sorted(
            [item for item in os.listdir(root_dir) if not item.endswith('.zip')]
        )
    return subfolders_dict

# 替换为你的大文件夹路径
root_directory = '/data/guohua/BeiJing/data/Nfs/'
result = list_subfolders_nfs(root_directory)
# print(result)

# 将结果字典写入JSON文件
json_filename = './nfs.json'  # 你可以根据需要修改文件名
with open(json_filename, 'w') as json_file:
    json.dump(result, json_file, indent=4)  # indent=4 用于格式化输出，使JSON文件易于阅读


# UAV
def list_subfolders_lasot(root_dir):
    subfolders_dict = {}
    class_sample = os.listdir(root_dir)
    for class_name in class_sample:
        subfolder_path = os.path.join(root_dir, class_name)
        subfolders_dict[class_name] = sorted(os.listdir(subfolder_path))
    return subfolders_dict

# 替换为你的大文件夹路径
root_directory = '/data/guohua/BeiJing/data/UAV123/UAV_Lasot_struc/'
result = list_subfolders_lasot(root_directory)
# print(result)

# 将结果字典写入JSON文件
json_filename = './uav.json'  # 你可以根据需要修改文件名
with open(json_filename, 'w') as json_file:
    json.dump(result, json_file, indent=4)  # indent=4 用于格式化输出，使JSON文件易于阅读
