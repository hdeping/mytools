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
        x = self.conv(x)
        # batchnorm 
        x = self.bn(x)
        # 1d max pool 
        x = F.avg_pool1d(x,kernel_size=4)
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
        # get conv layers
        self.layer_num = 4
        chanels = [1,5,10,20,40,32,40,40,40]
        for i in range(self.layer_num):
            self.layer_conv.add_module('conv'+str(i),baseConv1d(chanels[i],chanels[i+1],3,1,1))


        self.layer1 = nn.Sequential()
        self.layer1.add_module('gru', nn.GRU(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('batchnorm', nn.BatchNorm1d(self.hidden_dim))
        self.layer2.add_module('linear', nn.Linear(self.hidden_dim, self.bn_dim))
        # self.layer2.add_module('Sigmoid', nn.Sigmoid())

        self.layer3 = nn.Sequential()
        self.layer3.add_module('batchnorm', nn.BatchNorm1d(self.bn_dim))
        self.layer3.add_module('linear', nn.Linear(self.bn_dim, self.output_dim))

    def forward(self, x, mask, target):
        batch_size, fea_frames, fea_dim = x.size()
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


        out_target = out_target.contiguous().view(batch_size, fea_frames, -1)
        mask = mask.contiguous().view(batch_size, fea_frames, 1).expand(batch_size, fea_frames, out_target.size(2))
        out_target_mask = out_target * mask
        out_target_mask = out_target_mask.sum(dim=1)/mask.sum(dim=1)
        predict_target = F.softmax(out_target_mask, dim=1)

        # 计算loss
        tar_select_new = torch.gather(predict_target, 1, target)
        ce_loss = -torch.log(tar_select_new) 
        ce_loss = ce_loss.sum() / batch_size

        # 计算acc
        (data, predict) = predict_target.max(dim=1)
        predict = predict.contiguous().view(-1,1)
        correct = predict.eq(target).float()       
        num_samples = predict.size(0)
        sum_acc = correct.sum().item()
        acc = sum_acc/num_samples

        return acc, ce_loss
