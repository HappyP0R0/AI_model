import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

MAX_STEP = 5  # Max distance from img to img, stop searching if next 5 images doesn't exist
MAX_FRAMES = 5  # Max frames in one slide
path_to_folder = "E:\\AI_Data\\100Drivers\\data-splits\\Cross-modality-setting\\D1_to_N1\\"
image_folder_path = "E:\\AI_Data\\100Drivers\\Day\\Cam1\\"
file = "D1_train.txt"


class DataSetPrep:
    def __init__(self, txt_path, image_path):
        self.data = pd.read_csv(txt_path, sep='\t', header=None, names=['ID', 'ImagePath', 'Label'])
        self.image_path = image_path

    # Check if this is the first frame of this movement, it works because of this datasets file struct
    def is_first_frame(self, start_frame_path, image_path):
        is_first = 0
        is0 = 0

        if os.path.exists(image_path + start_frame_path):
            is_first = 1

        dir_name, file_name = os.path.split(start_frame_path)
        file_id = file_name.split('_')[-1].split('.')[0]  # Extract the frame number from file name

        if file_id[-1] == "a":
            return

        if (file_id[0] == '0'):
            is0 = 1

        file_id = int(file_id)
        result = file_name.split('.')[0]
        prefix = '_'.join(result.split('_')[:-1]) + '_'

        counter = 1
        while counter < MAX_STEP:
            curent_frame_num = f"{file_id - counter}.jpg"

            if (is0):
                curent_frame_num = '0' + curent_frame_num

            current_frame = prefix + curent_frame_num  # Adjust the format to fit file name format
            frame_path = dir_name + "/" + current_frame

            if os.path.exists(image_path + frame_path):
                is_first = 0
                break

            counter += 1

        return is_first

    def get_next_frame(self, last_frame):
        is0 = 0

        dir_name, file_name = os.path.split(last_frame)
        file_id = file_name.split('_')[-1].split('.')[0]  # Extract the frame number from file name

        if file_id[0] == '0':
            is0 = 1

        file_id = int(file_id)
        result = file_name.split('.')[0]
        prefix = '_'.join(result.split('_')[:-1]) + '_'

        final_frame_path = 0

        counter = 1
        while counter <= MAX_STEP:
            curent_frame_num = f"{file_id + counter}.jpg"

            if (is0):
                curent_frame_num = '0' + curent_frame_num

            current_frame = prefix + curent_frame_num  # Adjust the format to fit file name format
            frame_path = dir_name + "/" + current_frame
            if os.path.exists(self.image_path + frame_path):
                final_frame_path = frame_path
                break

            counter += 1

        return final_frame_path

    def list_of_start_frames(self):
        frames = {"ImagePath": [], "Label": []}

        for idx in range(len(self.data["ID"])):
            row = self.data.iloc[idx]
            is_first = self.is_first_frame(row["ImagePath"], self.image_path)
            if is_first:
                frames["ImagePath"].append(row["ImagePath"])
                frames["Label"].append(row["Label"])

        frames = pd.DataFrame(frames)
        return frames

    def get_slides(self, start_frames):
        frames = {"ImagePath": [], "Label": []}

        for idx in range(len(start_frames["ImagePath"])):
            row = start_frames.iloc[idx]
            start_frame = row["ImagePath"]
            label = row["Label"]

            cur_frame = start_frame  # Get an array of all video frames
            image_array = []
            while 1:
                image_array.append(cur_frame)
                next_frame = self.get_next_frame(cur_frame)
                if next_frame == 0:
                    break
                else:
                    cur_frame = next_frame

            for i in range(len(image_array) - 4):
                slide_image_array = []
                for j in range(MAX_FRAMES):
                    slide_image_array.append(image_array[i + j])

                frames["ImagePath"].append(slide_image_array)
                frames["Label"].append(label)

        frames = pd.DataFrame(frames)
        return frames


class VideoDataSet(Dataset):
    def __init__(self, frames, image_path, transform):
        self.data = frames
        self.transform = transform
        self.image_path = image_path

    def load_frames(self, frame_row):
        frames = []
        for i in range(len(frame_row)):
            frame_path = self.image_path + frame_row[i]
            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert("RGB")  # Use PIL to load the image
                frames.append(self.transform(img))
            else:
                print(f"Frame {frame_path} not found.")

        return torch.stack(frames)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        frames = self.load_frames(row['ImagePath'])
        label = torch.tensor(row['Label'], dtype=torch.long)
        return frames, label