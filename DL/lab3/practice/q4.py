import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__() #imp part
        self.w =nn.Parameter(torch.tensor(1.0))  #its nn not torch
        self.b =nn.Parameter(torch.tensor(1.0))  #notice no need to wite req grad

    def forward(self, x):
        return x*self.w + self.b

class LRDataset(Dataset):
    def __init__(self, x, y):
        self.x=x
        self.y=y

    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):  #get by index
        return self.x[idx], self.y[idx]

x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])
# learning_rate = torch.tensor(0.001)

dataset = LRDataset(x,y)
dataloader = DataLoader(dataset, batch_size=1, shuffle= True)  #imp
model = RegressionModel()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)  #imp to give criterion and optimizer

for epoch in range(100):
    ep_loss = 0.0

    for input, target in dataloader:
        optimizer.zero_grad()  #imp
        y_pred= model.forward(input)
        loss = criterion(target,y_pred)

        loss.backward()
        optimizer.step()
        ep_loss += loss.item()


    print(f'w:{model.w}, b:{model.b})')
