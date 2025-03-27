import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


X = torch.tensor([1, 5, 10, 10, 25, 50, 70, 75, 100], dtype=torch.float32).view(-1, 1)
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32).view(-1, 1)

class RegressionModel(nn.Module):

    def __init__(self, inShape, outShape):
        super().__init__()
        self.linear = nn.Linear(inShape, outShape)

    def forward(self, X):
        return self.linear(X)

model= RegressionModel(X.shape[1], y.shape[1])
print(model.state_dict())

criterion = nn.BCEWithLogitsLoss()
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
test= torch.tensor([40], dtype=torch.float32)

def classify(prob):
    if prob< 0.5:
        return 0
    return 1

with torch.inference_mode():
    y_pred = model(test)
    sigmoid_y_pred = torch.sigmoid(y_pred)
    print(f'X: {test}, y_pred {sigmoid_y_pred.detach().numpy()}, class: {classify(sigmoid_y_pred)}')

plt.plot(loss_list)
plt.show()