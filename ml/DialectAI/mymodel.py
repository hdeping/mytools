# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class LanNet(nn.Module):
    def __init__(self, input_dim_fb=40, input_dim_phoneme=8,hidden_dim=2048, bn_dim=100, output_dim=10):
        super(LanNet, self).__init__()
        self.input_dim_fb = input_dim_fb
        self.input_dim_phoneme = input_dim_phoneme
        self.hidden_dim = hidden_dim
        self.bn_dim = bn_dim
        self.output_dim = output_dim

        self.layer_fb = nn.Sequential()
        self.layer_fb.add_module('gru', nn.GRU(self.input_dim_fb, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False))
        self.layer_phoneme = nn.Sequential()
        self.layer_phoneme.add_module('gru', nn.GRU(self.input_dim_phoneme, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('batchnorm', nn.BatchNorm1d(2*self.hidden_dim))
        self.layer2.add_module('linear', nn.Linear(2*self.hidden_dim, self.bn_dim))
        # self.layer2.add_module('Sigmoid', nn.Sigmoid())

        self.layer3 = nn.Sequential()
        self.layer3.add_module('batchnorm', nn.BatchNorm1d(self.bn_dim))
        self.layer3.add_module('linear', nn.Linear(self.bn_dim, self.output_dim))

    def forward(self, src_fb, src_phoneme,mask_fb,mask_phoneme, target):
        batch_size, fea_frames_fb, fea_dim = src_fb.size()
        batch_size, fea_frames_phoneme, fea_dim = src_phoneme.size()

        # get gru output
        out_hidden_fb, hidd = self.layer_fb(src_fb)
        out_hidden_phoneme, hidd = self.layer_phoneme(src_phoneme)
        # summation of the two hidden states in the same node
        # out_hidden = out_hidden[:,:,0:self.hidden_dim] + out_hidden[:,:,self.hidden_dim:]
        #print(out_hidden.shape)
        # get  masked outputs
        # fb
        mask_fb = mask_fb.contiguous()
        mask_fb = mask_fb.view(batch_size, fea_frames_fb, 1)
        mask_fb = mask_fb.expand(batch_size, fea_frames_fb, out_hidden_fb.size(2))
        # phoneme
        #print("hidden")
        #print(out_hidden_fb.shape)
        #print(out_hidden_phoneme.shape)
        #print("mask")
        #print(mask_fb.shape)
        #print(mask_phoneme.shape)
        mask_phoneme = mask_phoneme.contiguous()
        mask_phoneme = mask_phoneme.view(batch_size, fea_frames_phoneme, 1)
        mask_phoneme = mask_phoneme.expand(batch_size, fea_frames_phoneme, out_hidden_phoneme.size(2))

        # output with new size (batch_size, hidden_dim)
        # fb
        out_hidden_fb = out_hidden_fb*mask_fb
        out_hidden_fb = out_hidden_fb.sum(dim=1)/mask_fb.sum(dim=1)
        # phoneme
        out_hidden_phoneme = out_hidden_phoneme*mask_phoneme
        out_hidden_phoneme = out_hidden_phoneme.sum(dim=1)/mask_phoneme.sum(dim=1)

        # concanate fb and phoneme
        out_hidden = torch.cat((out_hidden_fb,out_hidden_phoneme),dim=1)

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
        predict = predict.contiguous().view(-1,1)
        correct = predict.eq(target).float()       
        num_samples = predict.size(0)
        sum_acc = correct.sum().item()
        acc = sum_acc/num_samples

        return acc, ce_loss
