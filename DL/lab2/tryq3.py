import torch

x=torch.tensor([2.0], requires_grad=True)
b=torch.tensor([3.0], requires_grad=True)
w=torch.tensor([4.0], requires_grad=True)
u = w*x
v = u+b
a = torch.sigmoid(v)
a.backward()

print(f'da/dx={x.grad}')
print(f'da/db={b.grad}')
print(f'da/dw={w.grad}')

du_dx= w
du_dw= x
dv_db= 1
dv_du= 1
da_dv= a*(1-a)

da_dx= da_dv*dv_du*du_dx
print(f'da/dx={da_dx}')

da_db= da_dv*dv_db
print(f'da/db={da_db}')