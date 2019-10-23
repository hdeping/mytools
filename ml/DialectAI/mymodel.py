# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class baseConv1d(nn.Module):
    def __init__(self,input_chanel,output_chanel,kernel_size,stride,padding):
        super(baseConv1d,self).__init__()
        # architeture of the base conv1d
        self.conv = nn.Conv1d(input_chanel,output_chanel,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn   = nn.BatchNorm1d(output_chanel)
    def forward(self,x):
        # conv 
        #print(x.size())
        x = self.conv(x)
        # batchnorm 
        x = self.bn(x)
        # 1d avg pool 
        x = F.max_pool1d(x,kernel_size=2)
        # relu output
        x = F.relu(x)
        return x

class LanNet(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=2048, bn_dim=100, output_dim=10):
        super(LanNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bn_dim = bn_dim
        self.output_dim = output_dim

        self.layer_conv = nn.Sequential()
        chanels = [1,5,10,20,40]
        #chanels = [1,2,4,8,16,32,40,64,80]
        self.layer_num = 4
        for i in range(self.layer_num):
            self.layer_conv.add_module('conv'+str(i),baseConv1d(chanels[i],chanels[i+1],3,2,1))

        self.layer1 = nn.Sequential()
        self.layer1.add_module('gru', nn.GRU(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('batchnorm', nn.BatchNorm1d(self.hidden_dim))
        self.layer2.add_module('linear', nn.Linear(self.hidden_dim, self.bn_dim))
        # self.layer2.add_module('Sigmoid', nn.Sigmoid())

        self.layer3 = nn.Sequential()
        self.layer3.add_module('batchnorm', nn.BatchNorm1d(self.bn_dim))
        self.layer3.add_module('linear', nn.Linear(self.bn_dim, self.output_dim))

    def forward(self, x):
        batch_size, fea_frames, fea_dim = x.size()
        #print(x.size)
        # reshape the input
        x = x.contiguous().view(batch_size*fea_frames,1,-1)
        # conv layer
        x = self.layer_conv(x)
        # reshape x
        x = x.contiguous().view(batch_size,fea_frames,-1)

        # RNN layer
        out_hidden, hidd = self.layer1(x)
        #print(out_hidden.data.shape)
        out_hidden = out_hidden.contiguous().view(-1, out_hidden.size(-1))   
        #print(out_hidden.data.shape)
        out_bn = self.layer2(out_hidden)
        out_target = self.layer3(out_bn)

        return out_target
