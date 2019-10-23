# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class LanNet(nn.Module):
    def __init__(self, input_dim=40, pitch_dim=13,hidden_dim=2048, bn_dim=100, output_dim=10,alpha=0.5):
        super(LanNet, self).__init__()
        self.pitch_dim  = pitch_dim
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.bn_dim     = bn_dim
        self.output_dim = output_dim
        self.alpha = alpha

        # for pitch flux
        self.layer0 = nn.Sequential()
        self.layer0.add_module('gru', nn.GRU(self.pitch_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False))
        # for fbank
        self.layer1 = nn.Sequential()
        self.layer1.add_module('gru', nn.GRU(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('batchnorm', nn.BatchNorm1d(self.hidden_dim))
        self.layer2.add_module('linear', nn.Linear(self.hidden_dim, self.bn_dim))
        # self.layer2.add_module('Sigmoid', nn.Sigmoid())

        self.layer3 = nn.Sequential()
        self.layer3.add_module('batchnorm', nn.BatchNorm1d(self.bn_dim))
        self.layer3.add_module('linear', nn.Linear(self.bn_dim, self.output_dim))
    def getLoss(self,out_hidden,mask,target,fea_frames):
        batch_size = out_hidden.size(0)
        # get  masked outputs
        mask = mask.contiguous().view(batch_size, fea_frames, 1).expand(batch_size, fea_frames, out_hidden.size(2))
        # output with new size (batch_size, hidden_dim)
        out_hidden = out_hidden*mask
        out_hidden = out_hidden.sum(dim=1)/mask.sum(dim=1)
        #out_hidden = out_hidden.contiguous().view(-1, out_hidden.size(-1))   
        # shared weights
        out_bn = self.layer2(out_hidden)
        out_target = self.layer3(out_bn)
        predict_target = F.softmax(out_target, dim=1)
        #print(predict_target.shape,target.shape)

        return predict_target

    def forward(self, src, mask, target):
        batch_size, fea_frames, fea_dim = src.size()

        # get gru output
        out_hidden_pitch,hidd = self.layer0(src[:,:,self.input_dim:])
        out_hidden_fb   ,hidd = self.layer1(src[:,:,:self.input_dim])
        # summation of the two hidden states in the same node
        # out_hidden = out_hidden[:,:,0:self.hidden_dim] + out_hidden[:,:,self.hidden_dim:]
        #print(out_hidden.shape)
        # get the loss
        predict_pitch = self.getLoss(out_hidden_pitch,mask,target,fea_frames)
        predict_fb    = self.getLoss(out_hidden_fb,mask,target,fea_frames)
        
        predict_target = predict_pitch*self.alpha + predict_fb*(1 - self.alpha)

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
