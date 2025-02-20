import torch.nn.functional as F
import torch.nn as nn
import torch

img = torch.rand(6,6)
kernel = torch.ones(3,3)
img = img.unsqueeze(dim=0).unsqueeze(dim=0)
kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0)

for stride in [1,2,3]:
    for padding in [0,1,2]:
        output = F.conv2d(img ,kernel ,padding=padding ,stride=stride )
        print(f'stride:{stride}, padding:{padding}, output shape: {output.shape}')

print('\n\n')
print("Q2.")
conv_layer = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3 ,stride=1,padding=0 )
output = conv_layer(img)
print(f"weights from conv2d: {conv_layer.weight.data}, o/p shape: {output.shape}")

kernel = torch.ones(3,3)
kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0)
output = F.conv2d(img ,conv_layer.weight.data ,padding=0 ,stride=1 )  #weight.data from conv layer
print(f"o/p shape: {output.shape}")