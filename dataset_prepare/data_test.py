import os
import numpy as np
import random

# Set the path to the directory containing npz files
npz_directory = "/home/medicaldata3/WYMData/datasets/data/Synapse/train_npz/"

# Get a list of npz files in the directory
npz_files = [f for f in os.listdir(npz_directory) if f.endswith('.npz')]

# Randomly select 20 files
selected_files = random.sample(npz_files, 20)

# Initialize a dictionary to store the max and min values for each file
values = {}

# Iterate over the selected files and calculate the max and min values
for file in selected_files:
    data_path = os.path.join(npz_directory, file)
    data = np.load(data_path)
    image = data['image']
    values[file] = {'max': image.max(), 'min': image.min()}

# Print the max and min values for each file
for file, val in values.items():
    print(f"File: {file}, Max: {val['max']}, Min: {val['min']}")
