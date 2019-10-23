# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class LanNet(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=2048, bn_dim=100, output_dim=10):
        super(LanNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bn_dim = bn_dim
        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(self.input_dim,self.hidden_dim,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(self.hidden_dim,self.hidden_dim,kernel_size=3,padding=1)

        self.layer2 = nn.Sequential()
        self.layer2.add_module('batchnorm', nn.BatchNorm1d(self.hidden_dim))
        self.layer2.add_module('linear', nn.Linear(self.hidden_dim, self.bn_dim))
        # self.layer2.add_module('Sigmoid', nn.Sigmoid())

        self.layer3 = nn.Sequential()
        self.layer3.add_module('batchnorm', nn.BatchNorm1d(self.bn_dim))
        self.layer3.add_module('linear', nn.Linear(self.bn_dim, self.output_dim))

    def forward(self, src, mask, target):
        batch_size, fea_frames, fea_dim = src.size()

        # conv layer 
        # transpose
        src = src.transpose(1,2)
        src = F.relu(self.conv1(src))
        src = F.relu(self.conv2(src))

        # transpose
        out_hidden = src.transpose(1,2)
        # get gru output
        # summation of the two hidden states in the same node
        # out_hidden = out_hidden[:,:,0:self.hidden_dim] + out_hidden[:,:,self.hidden_dim:]
        #print(out_hidden.shape)
        # get  masked outputs
        mask = mask.contiguous().view(batch_size, fea_frames, 1).expand(batch_size, fea_frames, out_hidden.size(2))
        # output with new size (batch_size, hidden_dim)
        out_hidden = out_hidden*mask
        out_hidden = out_hidden.sum(dim=1)/mask.sum(dim=1)
        #out_hidden = out_hidden.contiguous().view(-1, out_hidden.size(-1))   
        out_bn = self.layer2(out_hidden)
        out_target = self.layer3(out_bn)


        #out_target = out_target.contiguous().view(batch_size, fea_frames, -1)
        #mask = mask.contiguous().view(batch_size, fea_frames, 1).expand(batch_size, fea_frames, out_target.size(2))
        #out_target_mask = out_target * mask
        #out_target_mask = out_target_mask.sum(dim=1)/mask.sum(dim=1)
        predict_target = F.softmax(out_target, dim=1)
        #print(predict_target.shape,target.shape)

        # 计算loss
        tar_select_new = torch.gather(predict_target, 1, target)
        ce_loss = -torch.log(tar_select_new) 
        ce_loss = ce_loss.sum() / batch_size

        # 计算acc
        (data, predict) = predict_target.max(dim=1)
        prediction = predict
        predict = predict.contiguous().view(-1,1)
        correct = predict.eq(target).float()       
        num_samples = predict.size(0)
        sum_acc = correct.sum().item()
        acc = sum_acc/num_samples

        #return acc, ce_loss,prediction
        return acc, ce_loss
