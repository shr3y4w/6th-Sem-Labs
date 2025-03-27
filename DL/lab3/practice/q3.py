# Revise the linear regression model by defining a user defined class titled RegressionModel with two parameters w and b
# as its member variables. Define a constructor to initialize w and b with value 1. Define four member functions namely
# forward(x) to implement wx+b, update() to update w and b values, reset_grad() to reset parameters to zero,
# criterion(y, yp) to implement MSE Loss given the predicted y value yp and the target label y. Define an object of this
# class named model and invoke all the methods. Plot the graph of epoch vs loss by varying epoch to 100 iterations.
import torch

class RegressionModel():
    def __init__(self):
        self.w =torch.tensor(1.0, requires_grad=True)
        self.b =torch.tensor(1.0, requires_grad=True)

    def forward(self, x):
        return x*self.w + self.b

    def update(self):
        self.w -= 0.001* self.w.grad
        self.b -= 0.001 * self.b.grad

    def reset_grad(self):
        self.w.grad.zero_()
        self.b.grad.zero_()

def criterion(y_j,y_pred):
    return (y_j-y_pred)**2

x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])
# learning_rate = torch.tensor(0.001)

model = RegressionModel()
loss_list=[]

for epoch in range(1000):
    loss = 0.0

    for j in range(len(x)):
        y_pred= model.forward(x[j])
        loss += criterion(y[j],y_pred)

    loss=loss/len(x)
    loss_list.append(loss.item())
    loss.backward()

    with torch.no_grad():
        model.update()
    model.reset_grad()

print(f'w:{model.w}, b:{model.b})')
