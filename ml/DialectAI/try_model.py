from mymodel import baseConv1d
import torch

conv = []
chanels = [1,2,4,8,16,32,40,40,40]
# get conv layers
layer_num = 8
for i in range(layer_num):
    conv_layer = baseConv1d(chanels[i],chanels[i+1],3,1,1)
    conv.append(conv_layer)

arr = torch.rand(100,1,400)
for i in range(layer_num):
    arr = conv[i](arr)
    print(arr.shape)

