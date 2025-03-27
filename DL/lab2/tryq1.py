import torch

a=torch.tensor([2.0], requires_grad=True)
b=torch.tensor([3.0], requires_grad=True)
x = 2*a + 3*b
y = 5*a*a + 3*b*b*b
z = 2*x + 3*y

z.backward()
print(f'dz/da={a.grad}')
print(f'dz/db={b.grad}')

dz_dx= 2
dz_dy= 3
dx_da= 2
dx_db= 3
dy_da= 10*a
dy_db= 9*b*b

dz_da= dz_dx*dx_da + dz_dy*dy_da
print(f'dz/da={dz_da}')