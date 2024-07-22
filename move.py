""" 移动文件到指定文件夹 """
import os
import shutil
import re

# 定义文件路径
log_file_path = 'logDataError.txt' #保存要移动的文件名
base_dir_text = '/sata/public/yyqi/Dataset/OCEAN/text' #原始文件夹路径
base_dir_joint = '/sata/public/yyqi/Dataset/OCEAN/new_joints' 
base_dir_vecs = '/sata/public/yyqi/Dataset/OCEAN/new_joint_vecs' 
process_dir = '/sata/public/yyqi/Dataset/OCEAN/bin/' 

os.makedirs(process_dir, exist_ok=True)

# 读取 log.txt 文件内容
with open(log_file_path, 'r') as file:
    log_content = file.readlines()

# 提取符合特定格式的文件名
pattern = re.compile(r'G\d{3}T\d{3}A\d{3}R\d{3}P\d{1}\.txt')
file_names = [pattern.search(line).group() for line in log_content if pattern.search(line)]

# 移动文件到 process 文件夹
for file_name in file_names:
    src_path_text = os.path.join(base_dir_text, file_name)
    src_path_joint = os.path.join(base_dir_joint, file_name[:-4]+'.npy')
    src_path_vecs = os.path.join(base_dir_vecs, file_name[:-4]+'.npy')
    dst_path = process_dir
    if os.path.exists(src_path_text):
        shutil.move(src_path_text, os.path.join(dst_path,'text'))
    if os.path.exists(src_path_joint):
        shutil.move(src_path_joint, os.path.join(dst_path,'new_joints'))
    if os.path.exists(src_path_vecs):
        shutil.move(src_path_vecs, os.path.join(dst_path,'new_joint_vecs'))
       
