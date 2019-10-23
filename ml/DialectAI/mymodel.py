# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class baseConv1d(nn.Module):
    def __init__(self,input_chanel,output_chanel,kernel_size,stride):
        super(baseConv1d,self).__init__()
        # architeture of the base conv1d
        self.conv = nn.Conv1d(input_chanel,output_chanel,kernel_size=kernel_size,stride=stride)
        self.bn   = nn.BatchNorm1d(output_chanel)
    def forward(self,x):
        # conv 
        x = self.conv(x)
        # batchnorm 
        x = self.bn(x)
        # 1d max pool 
        x = F.avg_pool1d(x,kernel_size=2)
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

        # 400-199-99
        self.conv1 = baseConv1d(1,4,3,2)
        # 99-49-24
        self.conv2 = baseConv1d(4,10,3,2)
        # 24-11-5
        self.conv3 = baseConv1d(10,20,3,2)
        # 5-2-1
        self.conv4 = baseConv1d(20,40,3,2)

        # gru layer
        #self.layer1 = nn.Sequential()
        self.layer1 = nn.GRU(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        # flatten the parameters
        #self.layer1.flatten_parameters()

        self.layer2 = nn.Sequential()
        self.layer2.add_module('batchnorm', nn.BatchNorm1d(self.hidden_dim))
        self.layer2.add_module('linear', nn.Linear(self.hidden_dim, self.bn_dim))
        # self.layer2.add_module('Sigmoid', nn.Sigmoid())

        self.layer3 = nn.Sequential()
        self.layer3.add_module('batchnorm', nn.BatchNorm1d(self.bn_dim))
        self.layer3.add_module('linear', nn.Linear(self.bn_dim, self.output_dim))

    def forward(self, x):
        #self.layer1.flatten_parameters()
        batch_size, fea_frames, fea_dim = x.size()
        # reshape the input
        x = x.contiguous().view(batch_size*fea_frames,1,-1)
        # conv layer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # reshape x
        #self.layer1.flatten_parameters()
        x = x.contiguous().view(batch_size,fea_frames,-1)

        # RNN layer
        out_hidden, hidd = self.layer1(x)
        #print(out_hidden.data.shape)
        out_hidden = out_hidden.contiguous().view(-1, out_hidden.size(-1))   
        #print(out_hidden.data.shape)
        out_bn = self.layer2(out_hidden)
        out_target = self.layer3(out_bn)


        return out_target
