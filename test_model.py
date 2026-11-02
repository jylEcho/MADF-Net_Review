import os
import re

# # 定义文件夹路径和输出文件路径
# data_path = '/home/image006/multi-phase/dataset_2d/dataset_v/train/'
# output_file_path = '/home/image006/multi-phase/multiphase-liverlesion/dataset_prepare/selected_patients.txt'
#
# def extract_lesion_numbers(data_path, output_file_path):
#     """提取文件夹中的Lesion-编号并保存到文本文件"""
#     lesion_numbers = set()
#     pattern = re.compile(r"([a-zA-Z]+)-(\d{3,4})-")
#
#     for file_name in os.listdir(data_path):
#         match = pattern.match(file_name)
#         if match:
#             lesion_number = match.group(0).rstrip('-')
#             lesion_numbers.add(lesion_number)
#
#     with open(output_file_path, 'w') as file:
#         for number in sorted(lesion_numbers):
#             file.write(number + '\n')
#
# extract_lesion_numbers(data_path, output_file_path)
import  numpy as np
import h5py
file_path = "/home/image006/multi-phase/dataset_2d/dataset_v/test/unknown-037-V.h5"
with h5py.File(file_path, 'r') as h5_file:
    keys = h5_file["box_keys"][:]
    values = h5_file["box_values"][:]
    crop_boxes = {key: value for key, value in zip(keys, values)}

print(crop_boxes)