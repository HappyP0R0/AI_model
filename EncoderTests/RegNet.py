import torch
from torchvision import transforms
from torchvision.models import regnet_y_400mf, RegNet_Y_400MF_Weights
from torch.utils.data import DataLoader
from torch import nn, optim
from PIL import Image

path_to_folder = "E:\AI_Data\\100Drivers\Day\Cam1\\"

# Custom Dataset class to load images and labels from a .txt file
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, txt_file, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        with open(txt_file, 'r') as f:
            for line in f:
                npk, path, label = line.strip().split()
                self.image_paths.append(path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = path_to_folder + self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Initialize data transforms (resizing to 224x224, normalizing)
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

# Load datasets using the custom dataset class
train_dataset = CustomImageDataset(txt_file=path_to_folder + 'driver_train.txt', transform=data_transforms['train'])
test_dataset = CustomImageDataset(txt_file=path_to_folder + 'driver_test.txt', transform=data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Initialize the RegNet model with pre-trained weights
regnet_model = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V1)

num_classes = 22
in_features = regnet_model.fc.in_features  # Access the input features of the final fully connected layer
regnet_model.fc = nn.Linear(in_features, num_classes)  # Replace the fully connected layer

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
regnet_model = regnet_model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(regnet_model.parameters(), lr=0.001)


# Training and testing loop
num_epochs = 10  # Set number of epochs


for epoch in range(num_epochs):
    # TRAINING PHASE
    regnet_model.train()  # Set the model to training mode

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear the gradients

        outputs = regnet_model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update weights

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    #print(f'Epoch {epoch}/{num_epochs - 1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # TESTING PHASE
    regnet_model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    test_corrects = 0

    with torch.no_grad():  # Disable gradient computation for testing
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = regnet_model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)

    test_loss = test_loss / len(test_dataset)
    test_acc = test_corrects.double() / len(test_dataset)

    print(f"Epoch {epoch}/{num_epochs - 1} Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
