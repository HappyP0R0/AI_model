import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Constants
MAX_STEP = 5
MAX_FRAMES = 5
IMG_SIZE = (224, 224)  # Image size for EfficientNet
BATCH_SIZE = 32
path_to_folder = "E:\AI_Data\\100Drivers\Day\Cam1\\"
training_file_path = path_to_folder + "driver_train.txt"
test_file_path = path_to_folder + "driver_test.txt"


# Function to read .txt file and process it
def load_video_data(txt_path):
    data = pd.read_csv(txt_path, sep='\t', header=None, names=['ID', 'ImagePath', 'Label'])
    return data


# Custom Dataset Class
class VideoDataset(Dataset):
    def __init__(self, txt_path):
        self.data = load_video_data(txt_path)
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.padding_value = 0

    def load_frames(self, start_frame_path):
        # Extract directory and file name of the first frame
        dir_name, file_name = os.path.split(start_frame_path)
        file_id = file_name.split('_')[-1].split('.')[0]  # Extract the frame number from file name

        result = file_name.split('.')[0]
        prefix = '_'.join(result.split('_')[:-1]) + '_'
        int_file_id = int(file_id)
        if file_id != str(int_file_id):
            prefix += '0'

        file_id = int_file_id

        frames = []
        counter = 0
        step = 0

        while len(frames) < MAX_FRAMES:
            current_frame = prefix + f"{file_id + counter:d}.jpg"  # Adjust the format to fit file name format
            frame_path = dir_name + "/" + current_frame

            if os.path.exists(frame_path):
                img = Image.open(frame_path).convert("RGB")  # Use PIL to load the image
                frames.append(self.transform(img))
                step = 0

            if step >= MAX_STEP:
                break

            counter += 1
            step += 1

        while len(frames) < MAX_FRAMES:
            # Create a zero tensor with the same shape as a single frame
            pad_frame = torch.zeros(3, *IMG_SIZE)  # shape: [3, 224, 224]
            frames.append(pad_frame)

        return torch.stack(frames)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        frames = self.load_frames(path_to_folder + row['ImagePath'])
        label = torch.tensor(row['Label'], dtype=torch.long)

        return frames, label


# Define the model architecture
class VideoClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(VideoClassifier, self).__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)  # EfficientNet backbone
        self.efficientnet.classifier = nn.Identity()  # Remove the classification layer
        self.lstm = nn.LSTM(input_size=1280, hidden_size=512, num_layers=1,
                            batch_first=True)  # Adjust input size for EfficientNet
        self.fc = nn.Linear(512, num_classes)  # Output layer

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)  # Reshape for EfficientNet
        x = self.efficientnet(x)  # Extract features
        x = x.view(batch_size, seq_len, -1)  # Reshape back to (batch_size, seq_len, features)
        x, _ = self.lstm(x)  # LSTM
        x = x[:, -1, :]  # Get the output of the last time step
        x = self.fc(x)  # Classify
        return x


# Load dataset
data = load_video_data(path_to_folder + 'driver_train.txt')
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Create Datasets and DataLoaders
train_dataset = VideoDataset(training_file_path)
val_dataset = VideoDataset(test_file_path)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VideoClassifier(num_classes=22).to(device)  # Adjust num_classes as needed
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 30

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for frames, labels in train_loader:
        frames, labels = frames.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(frames)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss for logging
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}")


# Validation Loop
model.eval()  # Set the model to evaluation mode
all_preds = []
all_labels = []
val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():  # Disable gradient calculation for validation
    for frames, labels in val_loader:
        frames, labels = frames.to(device), labels.to(device)

        # Forward pass (no gradient calculation)
        outputs = model(frames)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        # Get predicted labels
        _, predicted = torch.max(outputs.data, 1)

        # Save predictions and true labels
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate F1 score
f1 = f1_score(all_labels, all_preds, average='weighted')  # For multi-class classification

print(f'Validation Loss: {val_loss / len(val_loader):.4f}, '
      f'Validation Accuracy: {100 * correct / total:.2f}%, '
      f'F1 Score: {f1:.4f}')