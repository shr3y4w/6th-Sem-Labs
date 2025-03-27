import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

X1 = torch.tensor([3, 4, 5, 6, 2], dtype=torch.float32).view(-1, 1)
X2 = torch.tensor([8, 5, 7, 3, 1], dtype=torch.float32).view(-1, 1)
y = torch.tensor([-3.7, 3.5, 2.5, 11.5, 5.7],dtype=torch.float32).view(-1, 1)

X= torch.concatenate((X1,X2), dim=-1)  #last dim

class RegressionModel(nn.Module):

    def __init__(self, inShape, outShape):
        super().__init__()
        self.linear = nn.Linear(inShape, outShape)

    def forward(self, X):
        return self.linear(X)

model= RegressionModel(X.shape[1], y.shape[1])
print(model.state_dict())

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_list=[]

for epoch in range(1000):
    optimizer.zero_grad() #imp
    y_pred= model(X)
    loss = criterion(y_pred,y)
    loss_list.append(loss.detach().numpy())
    loss.backward()

    optimizer.step()
    if epoch % 100 ==0:
        print(f'epoch {epoch}, loss {loss.detach().numpy()}, w: {model.linear.weight.data.numpy()}, b: {model.linear.bias.data.numpy()}')

model.eval()
test= torch.tensor([2,1], dtype=torch.float32)

with torch.inference_mode():
    y_pred = model(test)
    print(f'X: {test}, y_pred {y_pred.detach().numpy()}')

plt.plot(loss_list)
plt.show()