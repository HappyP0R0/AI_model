import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import DatasetLoader as dl
import argparse
import time
import wandb

# Constants
BATCH_SIZE = 16

parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('-run_name', default=f'run_{time.time()}')
parser.add_argument('-epochs', default=10, type=int)
parser.add_argument('-use_pooling', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("-training_file", default="E:/AI_Data/100Drivers/Day/Cam1/driver_train.txt")
parser.add_argument("-test_file", default="E:/AI_Data/100Drivers/Day/Cam1/driver_test.txt")
parser.add_argument("-img_path_train", default="E:/AI_Data/100Drivers/Day/Cam1/")
parser.add_argument("-img_path_test", default="E:/AI_Data/100Drivers/Night/Cam1/")
args = parser.parse_args()

training_file_path = args.training_file
test_file_path = args.test_file
train_image_folder_path = args.img_path_train
test_image_folder_path = args.img_path_test
use_pooling = args.use_pooling


# Define the model architecture
class VideoClassifier(nn.Module):
    def __init__(self, num_classes=22):
        super(VideoClassifier, self).__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)  # EfficientNet backbone
        self.efficientnet.classifier = nn.Identity()  # Remove the classification layer
        self.lstm = nn.LSTM(input_size=1280, hidden_size=512, num_layers=1,
                            batch_first=True)  # Adjust input size for EfficientNet
        if use_pooling:
            self.fc = nn.Linear(1536, num_classes)  # Output layer with pooling
        else:
            self.fc = nn.Linear(512, num_classes)  # Output layer no pooling

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)  # Reshape for EfficientNet
        x = self.efficientnet(x)  # Extract features
        x = x.view(batch_size, seq_len, -1)  # Reshape back to (batch_size, seq_len, features)
        x, _ = self.lstm(x)  # LSTM

        if use_pooling:
            mean_pool = x.mean(dim=1)  # Mean
            max_pool = x.max(dim=1).values  # Max
            std_pool = x.std(dim=1)  # Standard deviation

            # Add together
            x = torch.cat([mean_pool, max_pool, std_pool], dim=1)  # (batch_size, hidden_size * 3)
        else:
            x = x[:, -1, :]  # Get the output of the last time step

        x = self.fc(x)  # Classify
        return x


# Create Datasets and DataLoaders
train_transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=(0.3, 1), contrast=(0.3, 1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

training_dataset_prep = dl.DataSetPrep(training_file_path, train_image_folder_path)
training_frames = training_dataset_prep.get_slides(training_dataset_prep.list_of_start_frames())
train_dataset = dl.VideoDataSet(training_frames, train_image_folder_path, train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# test dataset
test_transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset_prep = dl.DataSetPrep(test_file_path, test_image_folder_path)
test_frames = test_dataset_prep.get_slides(test_dataset_prep.list_of_start_frames())
test_dataset = dl.VideoDataSet(test_frames, test_image_folder_path, test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VideoClassifier(num_classes=22).to(device)  # Adjust num_classes as needed
criterion = nn.CrossEntropyLoss()
optimizer = optim.RAdam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = args.epochs

wandb.init(
    # set the wandb project where this run will be logged
    project="bachelors",
    name=args.run_name + str(time.time()),

    # track hyperparameters and run metadata
    config={
        "architecture": "EfficientNet + LSTM",
        "dataset": "100-drivers",
        "epochs": args.epochs,
        "learning_rate": 0.001,
        "device": 'CUDA' if torch.cuda.is_available() else 'CPU',
        "pooling": use_pooling
    }
)

class_labels = [
    "Safe Driving", "Sleeping", "Yawning",
    "Talking Left", "Talking Right", "Texting Left",
    "Texting Right", "Make Up", "Look Left", "Look Right",
    "Look Up", "Look Down", "Smoke Left",
    "Smoke Right", "Smoke Mouth", "Eat Left", "Eat Right",
    "Operate Radio", "Operate GPS", "Reach Behind",
    "Leave Steering Wheel", "Talk To Passenger"]

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    running_corrects = 0.0

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

        # Accumulate loss and acc for logging
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()

    epoch_acc = running_corrects / len(train_dataset)
    epoch_loss = running_loss / len(train_dataset)
    print(
        f'epoch: {epoch} train acc: {epoch_acc}, train loss: {epoch_loss}, corrects: {running_corrects}, loss: {epoch_loss}')

    # Validation Loop
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    val_loss = 0.0
    correct = 0.0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for validation
        for frames, labels in test_loader:
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
    test_loss = val_loss / total
    test_acc = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')  # For multi-class classification
    print(
        f'epoch: {epoch} test acc: {test_loss}, test loss: {test_acc}, f1: {f1}')

    wandb.log({
        "train_acc": epoch_acc,
        "train_loss": epoch_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": f1,
        "conf_mat_"+str(epoch): wandb.sklearn.plot_confusion_matrix(
            all_labels,
            all_preds,
            class_labels
        )
    }, epoch+1)

wandb.finish()
