import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
MNIST_train = torchvision.datasets.MNIST(root='.', train=True, download=True, transform=transform)
MNIST_test = torchvision.datasets.MNIST(root='.', train=False, download=True, transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader_train = DataLoader(dataset=MNIST_train, batch_size= 64, shuffle= True)
dataloader_test = DataLoader(dataset=MNIST_test, batch_size= 64, shuffle= False)  #batch size im giving for mnist

class FFNN(nn.Module):

    def __init__(self):
        super(FFNN, self).__init__()
        self.linear1 = nn.Linear(28*28,128, bias=True)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(128,64, bias=True)
        self.activation2 = nn.ReLU()
        self.output = nn.Linear(64, 10, bias=True)

    def forward(self,x):
        x= x.view(-1,28*28)   #im reshaping it
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.output(x)
        return x

model = FFNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
loss_list=[]

for epoch in range(10):
    tot_loss=0

    for ip, target in dataloader_train:

        #imp to send to device
        ip = ip.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        y_pred= model(ip)
        loss = criterion(y_pred, target)  #has to be item
        loss.backward()
        tot_loss+=loss.item()
        optimizer.step()

    tot_loss = tot_loss / len(dataloader_train)
    loss_list.append(tot_loss)

    print(f'epoch: {epoch+1}/10, loss: {tot_loss}')

for name, param in model.named_parameters():
    print(f'{name}: {param.data}')

params = sum( p.numel() for p in model.parameters() if p.requires_grad)  #imp, also numel()
print(f'Total number of parameters: {params}')

model.eval()

y_pred=[]
y_true=[]

with torch.inference_mode():
    for ip, target in dataloader_test:
        ip=ip.to(device)
        target = target.to(device)
        outputs= model(ip)
        _,y_p = torch.max(outputs,1)

        y_pred.extend(y_p.cpu().numpy())
        y_true.extend(target.cpu().numpy())

conf_matrix = confusion_matrix(np.array(y_true), np.array(y_pred))
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.plot(loss_list)
plt.show()

