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
class baseLinear(nn.Module):
    def __init__(self,input_chanel,output_chanel):
        super(baseLinear,self).__init__()

        self.fc = nn.Linear(input_chanel,output_chanel)
    def forward(self,x):
        x = self.fc(x)
        x = F.relu(x)
        return x


class LanNet(nn.Module):
    def __init__(self, input_dim=48,hidden_dim=1024):
        super(LanNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.layer = nn.Sequential()
        self.layer.add_module('fc1', baseLinear(self.input_dim,self.hidden_dim))
        self.layer.add_module('fc2', baseLinear(self.hidden_dim,self.hidden_dim))
        self.layer.add_module('fc3', baseLinear(self.hidden_dim,40))
        self.layer.add_module('fc4', baseLinear(40,self.hidden_dim))
        self.layer.add_module('fc5', baseLinear(self.hidden_dim,self.hidden_dim))
        self.layer.add_module('fc6', baseLinear(self.hidden_dim,self.input_dim))

    def forward(self, x):
        batch_size, fea_frames, fea_dim = x.size()
        #print(x.size)
        # reshape the input
        x = x.contiguous().view(batch_size*fea_frames,-1)
        out_target = self.layer(x)

        return out_target
