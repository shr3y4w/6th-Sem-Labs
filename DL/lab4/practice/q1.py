import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = (torch.tensor([0, 1, 1, 0], dtype=torch.float32).view(-1, 1))

class XORDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class XORModel(nn.Module):

    def __init__(self):
        super(XORModel, self).__init__()
        self.linear1 = nn.Linear(2,2, bias=True)
        self.activation1 = nn.Sigmoid()
        self.linear2 = nn.Linear(2,1, bias=True)
        self.activation2 = nn.Sigmoid()

    def forward(self,x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = XORDataset(X, Y)
dataloader = DataLoader(dataset, batch_size= 2, shuffle= True)
model = XORModel().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()
loss_list=[]

for epoch in range(10000):
    tot_loss=0

    for ip, target in dataloader:

        #imp to send to device
        ip = ip.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        y_pred= model(ip)
        loss = criterion(y_pred, target)  #has to be item
        loss.backward()
        tot_loss+=loss.item()
        optimizer.step()

    tot_loss = tot_loss / len(dataloader)
    loss_list.append(tot_loss)

    if epoch%1000==0:
        print(f'epoch: {epoch}/10000, loss: {tot_loss}')

for name, param in model.named_parameters():
    print(f'{name}: {param.data}')

params = sum( p.numel() for p in model.parameters() if p.requires_grad)  #imp, also numel()
print(f'Total number of parameters: {params}')

model.eval()
with torch.inference_mode():
    for ip in X:
        ip=ip.to(device)
        y_pred= model(ip)
        print(f'X: {ip.cpu().numpy()}, y_pred: {y_pred.cpu().numpy()}')  #needs to be cpu

plt.plot(loss_list)
plt.show()

