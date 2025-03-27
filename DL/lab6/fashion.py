import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import torchvision.datasets as datasets


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load FashionMNIST Test Data
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.net= nn.Sequential(nn.Conv2d(1,64,kernel_size=3),
                                nn.ReLU(),
                                nn.MaxPool2d((2,2), stride=2),
                                nn.Conv2d(64, 128, kernel_size=3),
                                nn.ReLU(),
                                nn.MaxPool2d((2, 2), stride=2),
                                nn.Conv2d(128, 64, kernel_size=3),
                                nn.ReLU(),
                                nn.MaxPool2d((2, 2), stride=2),
                                )
        self.classify_head = nn.Sequential(nn.Linear(64, 20, bias=True),
                                           nn.ReLU(),
                                           nn.Linear(20, 10, bias=True)
                                           )

    def forward(self, x):
        x = self.net(x)
        return self.classify_head(x.view(-1,64))


model = CNN()
model.load_state_dict( torch.load("./model.pt", map_location= device))
model.to(device)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Set Model to Evaluation Mode
model.eval()

correct = 0
total = 0

with torch.no_grad():  # No gradient computation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Get Predicted Class
        _, predicted = torch.max(outputs, 1)

        print("True Labels: ", labels.cpu().numpy())
        print("Predicted  : ", predicted.cpu().numpy())

        total += labels.size(0)
        correct += (labels==predicted).sum().item()
# Compute Accuracy
accuracy = 100.0 * correct / total
print(f"FashionMNIST Accuracy: {accuracy:.2f}%")

