import torch
from resnet import resnet18

model = resnet18()

x = torch.randn(10,1,2000,40)
y = model(x)

print(y.shape)
