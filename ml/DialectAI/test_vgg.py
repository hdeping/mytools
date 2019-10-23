import torch
from vgg2 import vgg13,vgg16,vgg19

x = torch.randn(16,1,101,40)
model = vgg19()
model.cuda()
x = x.cuda()
print(model)

print(x.shape)
y = model(x)
print(y.shape)

torch.save(model.state_dict(),"vgg.pt")
