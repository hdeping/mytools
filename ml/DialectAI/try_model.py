#from mymodel import baseConv1d
import torch.nn as nn
import torch

#conv = []
##chanels = [1,2,4,8,16,32,40,64,80]
#chanels = [1,5,10,20,40]
## get conv layers
#layer_num = 4
#for i in range(layer_num):
#    conv_layer = baseConv1d(chanels[i],chanels[i+1],3,2,1)
#    conv.append(conv_layer)
#
#arr = torch.rand(100,1,320)
#for i in range(layer_num):
#    arr = conv[i](arr)
#    print(arr.shape)
conv1 = nn.Conv1d(1, 10, kernel_size=3, stride=1, padding=1)
conv2 = nn.Conv1d(10, 20, kernel_size=2, stride=2, padding=0)
conv3 = nn.Conv1d(20, 40, kernel_size=3, stride=1, padding=1)
conv4 = nn.Conv1d(40, 40, kernel_size=3, stride=1, padding=1)
deconv1 = nn.ConvTranspose1d(40, 40, kernel_size=3, stride=1, padding=1)

arr = torch.rand(100,1,400)
x = conv1(arr)
print(x.shape)
x = conv2(x)
print(x.shape)
