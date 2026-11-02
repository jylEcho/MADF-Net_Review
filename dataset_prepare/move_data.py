import os
import re
import shutil
import random
import pandas as pd

# 定义路径
data_path = '/home/image006/multi-phase/data_nii.gz/'
mask_path = '/home/image006/multi-phase/mask_correction/'
nnunet_image_path = '/home/medicaldata3/WYMData/nnunet/nnUNet_raw/Dataset003_liver/imagesTs/'
nnunet_label_path = '/home/medicaldata3/WYMData/nnunet/nnUNet_raw/Dataset003_liver/labelsTs/'
output_csv_path = '/home/image006/multi-phase/multiphase-liverlesion/copied_files.csv'
lesion_names = ['HCC', 'FNH', 'ICC', 'LA', 'Lipoma', 'unknown']

# 选择20个病人
def select_patients():
    pattern = re.compile(r"([a-zA-Z]+)-(\d{3})-([ADV])\.nii\.gz")
    patients = set()

    for file_name in os.listdir(data_path):
        match = pattern.match(file_name)
        if match:
            patients.add(match.group(1) + '-' + match.group(2))

    return random.sample(patients, 20)

def copy_files(patients):
    """复制选中病人的文件并记录信息"""
    copied_files = []
    counter = 1
    for patient in patients:
        for phase in ['A', 'D', 'V']:
            new_id = f"Liver_{counter:03d}_0000"
            counter += 1

            # 复制data文件
            data_file = f"{patient}-{phase}.nii.gz"
            src_file = os.path.join(data_path, data_file)
            dst_file = os.path.join(nnunet_image_path, f"{new_id}.nii.gz")
            shutil.copy(src_file, dst_file)

            # 复制mask文件
            mask_file = data_file
            src_mask_file = os.path.join(mask_path, mask_file)
            dst_mask_file = os.path.join(nnunet_label_path, f"{new_id}.nii.gz")
            shutil.copy(src_mask_file, dst_mask_file)

            # 记录文件信息
            copied_files.append({
                'Original_File_Name': data_file,
                'New_File_Name': f"{new_id}.nii.gz",
                'Phase': phase
            })

    return copied_files

selected_patients = select_patients()
copied_data = copy_files(selected_patients)

# 创建DataFrame并保存为CSV
df = pd.DataFrame(copied_data)
df.to_csv(output_csv_path, index=False)
print("文件复制完成，并已生成CSV文件。")
