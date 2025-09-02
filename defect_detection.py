import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


good_images_dir = r'D:\project\project\THT IMAGES\good_images'
bad_images_dir = r'D:\project\project\THT IMAGES\dataset_devi\NOK'
reference_csv = r'D:\project\project\all datasets\filtered_dataset.csv'
cluster_dirs = [
    r'D:\project\project\folderfor testseperateprograms\meanshiftclusters\Cluster_0',
    r'D:\project\project\folderfor testseperateprograms\meanshiftclusters\Cluster_1',
    r'D:\project\project\folderfor testseperateprograms\meanshiftclusters\Cluster_2',
    r'D:\project\project\folderfor testseperateprograms\meanshiftclusters\Cluster_3',
    r'D:\project\project\folderfor testseperateprograms\meanshiftclusters\Cluster_4',
    r'D:\project\project\folderfor testseperateprograms\meanshiftclusters\Cluster_5',
    r'D:\project\project\folderfor testseperateprograms\meanshiftclusters\Cluster_6',
    r'D:\project\project\folderfor testseperateprograms\meanshiftclusters\Cluster_7',
    r'D:\project\project\folderfor testseperateprograms\meanshiftclusters\Cluster_8',
    r'D:\project\project\folderfor testseperateprograms\meanshiftclusters\Cluster_9'
]
test_set_dir = r'D:\project\project\THT IMAGES\testset\combined'


class DefectDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {img_path}, Error: {e}")
            image = Image.new('RGB', (224, 224))  
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


image_paths = []
labels = []

label_mapping = pd.read_csv(reference_csv).set_index('image_name')['label'].to_dict()


for img_name in os.listdir(good_images_dir):
    img_path = os.path.join(good_images_dir, img_name)
    if os.path.isfile(img_path):
        image_paths.append(img_path)
        labels.append(0)


for img_name in os.listdir(bad_images_dir):
    img_path = os.path.join(bad_images_dir, img_name)
    if os.path.isfile(img_path):
        image_paths.append(img_path)
        labels.append(1)


for cluster_dir in cluster_dirs:
    for img_name in os.listdir(cluster_dir):
        img_path = os.path.join(cluster_dir, img_name)
        if os.path.isfile(img_path):
            image_paths.append(img_path)
            labels.append(label_mapping.get(img_name, 1))  # Default to label 1 if not found in CSV


train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
val_paths, test_paths, val_labels, test_labels = train_test_split(val_paths, val_labels, test_size=0.5, random_state=42)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = DefectDataset(train_paths, train_labels, transform=transform)
val_dataset = DefectDataset(val_paths, val_labels, transform=transform)
test_dataset = DefectDataset(test_paths, test_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the ResNet 50 Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=None)  # Randomly initialized
model.fc = nn.Linear(model.fc.in_features, 2)  # Modify for binary classification
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Train the Model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_predictions += labels.size(0)

        scheduler.step()
        train_acc = correct_predictions.double() / total_predictions

        # Validation phase
        model.eval()
        val_correct, val_loss = 0, 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels)

        val_acc = val_correct.double() / len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader.dataset):.4f}, "
              f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25)

#  Evaluate the Model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Good', 'Bad'])
    disp.plot(cmap='Blues')
    plt.show()

# Evaluate the model
evaluate_model(model, test_loader)