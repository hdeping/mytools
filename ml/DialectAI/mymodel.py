# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class baseFC(nn.Module):
    def __init__(self,input_chanel,output_chanel):
        super(baseFC,self).__init__()
        # architeture of the base conv1d
        self.fc = nn.Linear(input_chanel,output_chanel)
        self.bn   = nn.BatchNorm1d(output_chanel)
    def forward(self,x):
        # conv 
        x = self.fc(x)
        # batchnorm 
        x = self.bn(x)
        # 1d avg pool 
        # x = F.avg_pool1d(x,kernel_size=2)
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

        self.layer_fc = nn.Sequential()
        self.layer_num = 2
        #chanels = [1,2,4,8,16,32,40,64,80]
        chanels = [400,100,40]
        for i in range(self.layer_num):
            self.layer_fc.add_module('fc'+str(i),baseFC(chanels[i],chanels[i+1]))

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
        x = x.contiguous().view(batch_size*fea_frames,-1)
        # fc layer
        x = self.layer_fc(x)
        # reshape the x
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
