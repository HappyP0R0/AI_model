import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

path_to_folder = "E:\AI_Data\\100Drivers\Day\Cam1\\"

class CustomImageDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Read the .txt file and extract image paths and labels
        with open(txt_file, 'r') as f:
            for line in f:
                npk, path, label = line.strip().split()
                self.image_paths.append(path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image from the file path
        img_path = path_to_folder + self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Get the label
        label = self.labels[idx]

        # Apply the transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label


# Define transforms for data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
train_dataset = CustomImageDataset(path_to_folder + 'driver_train', data_transforms['train'])
test_dataset = CustomImageDataset(path_to_folder + 'driver_test.txt', data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VGG16 pre-trained model
vgg_model = models.vgg16(weights=VGG16_Weights.DEFAULT)

# Freeze the pre-trained layers
for param in vgg_model.parameters():
    param.requires_grad = False

# Dataset classifiers
num_classes = 22  # 22 class in out dataset
vgg_model.classifier[6] = nn.Linear(4096, num_classes)

vgg_model = vgg_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg_model.classifier.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):

    vgg_model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = vgg_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    print(f'Epoch {epoch}/{num_epochs - 1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    vgg_model.eval()
    test_loss = 0.0
    test_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = vgg_model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)

    test_loss = test_loss / len(test_dataset)
    test_acc = test_corrects.double() / len(test_dataset)

    print(f'Epoch {epoch}/{num_epochs - 1} Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

print(f"Using device: {device}")  # Will print 'cuda' for GPU, 'cpu' otherwise
