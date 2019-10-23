
import torch
from resnet import resnet18
import time

model = resnet18()
print(model)

t1 = time.time()
#model = torch.nn.DataParallel(model)
model = model.cuda()
a = torch.randn(16,1,200,40)
print(a.shape)
a = a.cuda()
b = model(a)
print(b.shape)
t2 = time.time()
print("time",t2 - t1)
