# 生成训练数据的liverbox，
import os
import nibabel as nib
import numpy as np

def read_train_patients(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def calculate_bounding_box(mask, target_label):
    positions = np.where(mask == target_label)
    if positions[0].size == 0:
        return None
    min_x, max_x = np.min(positions[0]), np.max(positions[0])
    min_y, max_y = np.min(positions[1]), np.max(positions[1])
    return min_x, max_x, min_y, max_y

def calculate_liver_box(mask_path, patient):
    lesion, patient_number = patient.split('-')
    mask_file = os.path.join(mask_path, f"{lesion}-{patient_number}-V.nii.gz")
    mask_img = nib.load(mask_file)
    mask_slices = mask_img.get_fdata()

    min_x, max_x, min_y, max_y, min_z, max_z = [np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf]

    for z in range(mask_slices.shape[2]):
        slice_mask = mask_slices[:, :, z]
        box = calculate_bounding_box(slice_mask, 1) #label 1为liver
        if box is not None:
            min_x, max_x = min(min_x, box[0]), max(max_x, box[1])
            min_y, max_y = min(min_y, box[2]), max(max_y, box[3])
            min_z, max_z = min(min_z, z), max(max_z, z)

    if min_x == np.inf:
        return None  # 没有找到肝脏

    return min_x, max_x, min_y, max_y, min_z, max_z

# 定义路径
train_patients_path = '/root/autodl-tmp/train_patients.txt' #'/home/image006/multi-phase/multiphase-liverlesion/dataset_prepare/train_patients.txt'
mask_path = '/root/autodl-tmp/mask_correction' #'/home/image006/multi-phase/mask_correction/'
output_file_path = '/root/autodl-tmp/liver_boxes.txt' #'/home/image006/multi-phase/multiphase-liverlesion/dataset_prepare/liver_boxes.txt'

# 读取训练病人列表
train_patients = read_train_patients(train_patients_path)

# 计算每个病人的liver box
with open(output_file_path, 'w') as output_file:
    for patient in train_patients:
        liver_box = calculate_liver_box(mask_path, patient)
        if liver_box is not None:
            output_file.write(f"{patient} {' '.join(map(str, liver_box))}\n")
        else:
            raise ValueError(f"没有找到肝脏，病人：{patient}")
print(f"Liver box计算完成，结果已保存到{output_file_path}")

