import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

MAX_STEP = 5
IMG_SIZE = (224, 224)  # Image size for EfficientNet
BATCH_SIZE = 32
path_to_folder = "E:\AI_Data\\100Drivers\Day\Cam1\\"
training_file_path = path_to_folder + "driver_training.txt"
test_file_path = path_to_folder + "driver_test.txt"


def load_video_data(txt_path):
    data = pd.read_csv(txt_path, sep='\t', header=None, names=['ID', 'ImagePath', 'Label'])
    return data


class VideoDataSet(Dataset):
    def __init__(self, txt_path):
        self.data = load_video_data(txt_path)
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_frames(self, start_frame_path):
        # Extract directory and file name of the first frame
        is0 = 0
        dir_name, file_name = os.path.split(start_frame_path)
        file_id = file_name.split('_')[-1].split('.')[0]  # Extract the frame number from file name
        if (file_id[0] == '0'):
            is0 = 1
        file_id = int(file_id)
        result = file_name.split('.')[0]
        prefix = '_'.join(result.split('_')[:-1]) + '_'

        frames = []
        counter = 0
        step = 0
        while 1:
            curent_frame_num = f"{file_id + counter:07d}.jpg"
            if (is0):
                curent_frame_num = '0' + curent_frame_num
            current_frame = prefix + curent_frame_num  # Adjust the format to fit file name format
            frame_path = dir_name + "/" + current_frame

            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert("RGB")  # Use PIL to load the image
                frames.append(self.transform(img))
                step = 0
            else:
                print(f"Frame {frame_path} not found.")

            if step >= MAX_STEP:
                break

            counter += 1
            step += 1

        return torch.stack(frames)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        frames = self.load_video_frames(row['ImagePath'])
        label = torch.tensor(row['Label'], dtype=torch.long)

        return frames, label


videodataset = VideoDataSet(test_file_path)
videodataset.load_frames(path_to_folder + "C1_Drive_Safe/P024_V1_S1_0102080228.jpg")
