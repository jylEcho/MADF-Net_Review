import os
import numpy as np

def check_npz_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.npz'):
            filepath = os.path.join(directory, filename)
            try:
                np.load(filepath, allow_pickle=True)
                # print(f"文件 {filename} 加载成功")
            except Exception as e:
                print(f"加载文件 {filename} 时出错: {e}")

# 替换为您的目录路径
data_path = '/data/qklee/liver/process_data/train/ABS-147-V-0034.npz'
data = np.load(data_path)
print(data.shape)
