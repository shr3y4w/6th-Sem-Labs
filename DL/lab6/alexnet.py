import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader

import zipfile

dataset_path = "cats_and_dogs_filtered.zip"
extract_path = "."

with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),           # Convert image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet mean and std
])


data_dir = "cats_and_dogs_filtered"

train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "validation"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


alexnet = models.alexnet(pretrained=True)
num_features = alexnet.classifier[6].in_features  # Get the input features of the last layer
alexnet.classifier[6] = nn.Linear(num_features, 2)  # Change the output to 2 classes

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet.parameters(), lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alexnet.to(device)

num_epochs = 5  # You can increase this

for epoch in range(num_epochs):
    alexnet.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = alexnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

alexnet.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = alexnet(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total:.2f}%")

import random

# Get a batch from validation set
dataiter = iter(val_loader)
images, labels = next(dataiter)

# Get model predictions
outputs = alexnet(images.to(device))
_, predicted = torch.max(outputs, 1)

# Map class indices to labels
class_names = train_dataset.classes  # ['cats', 'dogs']

# Plot images with predicted labels
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
axes = axes.flatten()

for i in range(16):
    img = images[i].permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Unnormalize
    img = np.clip(img, 0, 1)

    axes[i].imshow(img)
    axes[i].set_title(f"Pred: {class_names[predicted[i]]}")
    axes[i].axis("off")

plt.show()

