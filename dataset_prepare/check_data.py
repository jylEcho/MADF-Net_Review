import os
import re
import shutil
## 检查是否3期都存在
# 定义路径
data_path = 'G:/total_data/data_nii.gz/'
mask_path = 'G:/total_data/mask_correction/'
doubt_data_path = 'G:/total_data/doubt_data/'
doubt_label_path = 'G:/total_data/doubt_label/'
lesion_names = ['HCC', 'FNH', 'ICC', 'LA', 'Lipoma', 'unknown']

def move_files(file_name, src_path, dst_path):
    """移动文件到指定目录"""
    src_file = os.path.join(src_path, file_name)
    dst_file = os.path.join(dst_path, file_name)
    if os.path.exists(src_file):
        shutil.move(src_file, dst_file)
    else:
        print(f"文件 {src_file} 不存在，无法移动。")

def check_and_move_patient_files():
    """检查每个病人是否都有三个期的文件，并移动有问题的文件"""
    lesion_pattern = "|".join(lesion_names)
    pattern = re.compile(rf"({lesion_pattern})-(\d{{3}})-([ADV])\.nii\.gz")
    patient_files = {}

    # 遍历文件并分类
    for file_name in os.listdir(data_path):
        match = pattern.match(file_name)
        if match:
            lesion, patient_id, phase = match.groups()
            key = (lesion, patient_id)
            patient_files.setdefault(key, set()).add(phase)

    # 检查每个病人的文件是否齐全并移动不完整的文件
    for key, phases in patient_files.items():
        if len(phases) != 3:
            print(f"病人 {key[1]}（病灶：{key[0]}）的文件不齐全，正在移动...")
            for phase in phases:
                file_name = f"{key[0]}-{key[1]}-{phase}.nii.gz"
                move_files(file_name, data_path, doubt_data_path)
                move_files(file_name, mask_path, doubt_label_path)

check_and_move_patient_files()
print("文件检查和移动完成。")
