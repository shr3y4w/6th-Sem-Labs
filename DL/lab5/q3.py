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

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
train_data = torchvision.datasets.MNIST(root='.', transform= transforms, download= True, train= True)
test_data = torchvision.datasets.MNIST(root='.', transform= transforms, download= True, train= False)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

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

model=CNN()
device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer= torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    tot_loss = 0

    for ip, target in train_loader:
        ip = ip.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(ip)
        loss= criterion(output,target)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    print(f'epoch {epoch}, total loss: {tot_loss}')

y_true=[]
y_pred=[]

model.eval()
with torch.inference_mode():
    for ip, target in test_loader:
        ip = ip.to(device)
        target = target.to(device)
        yp= model(ip)
        _,pred = torch.max(yp.data, 1)

        y_true.extend(target.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

conf_matrix = confusion_matrix(np.array(y_true), np.array(y_pred))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels= range(10), yticklabels= range(10))
plt.show()

total_params= sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'total params: {total_params}')

print("Accuracy score of the model: ",accuracy_score(y_true, y_pred))