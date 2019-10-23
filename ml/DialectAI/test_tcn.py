from tcn import TemporalConvNet
import torch

conv1 = TemporalConvNet(40,[40,64,80,128])

x = torch.randn(64,40,1280)
y = conv1(x)
print(y.shape)
