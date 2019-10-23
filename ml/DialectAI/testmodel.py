# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np


from resnet import resnet18



class LanNet(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=2048, bn_dim=100, output_dim=10):
        super(LanNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bn_dim = bn_dim
        self.output_dim = output_dim

        self.conv = resnet18()

        self.layer_gru = nn.Sequential()
        self.layer_gru.add_module('gru', nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True))

        self.layer1 = nn.Sequential()
        self.layer1.add_module('batchnorm', nn.BatchNorm1d(self.hidden_dim))
        self.layer1.add_module('linear', nn.Linear(self.hidden_dim, self.bn_dim))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('batchnorm', nn.BatchNorm1d(self.bn_dim))
        self.layer2.add_module('linear', nn.Linear(self.bn_dim, self.output_dim))

    def getBiHidden(self,layer,src,frames):
        # pack the sequence
        src = pack_padded_sequence(src,frames,batch_first=True)
        # get the gru output
        out_hidden, hidd = layer(src)
        # unpack the sequence
        out_hidden,lengths = pad_packed_sequence(out_hidden,batch_first=True)
        # add the forward-backward value
        out_hidden = out_hidden[:,:,0:self.hidden_dim] + out_hidden[:,:,self.hidden_dim:]
        return out_hidden

    def forward(self, x, frames,target):
        batch_size, fea_frames, fea_dim = x.size()
        # squeeze frames:  [batch_size,1] --> [batch_size]
        frames = frames.squeeze()
        # get packed sequence
        sorted_frames,sorted_indeces = torch.sort(frames,descending=True)
        # new input 
        x = x[sorted_indeces]
        # conv output
        # new target
        target = target[sorted_indeces]

        x = x.unsqueeze(1)
        x = self.conv(x)

        # squeeze
        # B,F,T -> B,T,F
        x = x.squeeze()
        x = x.transpose(1,2)

        sorted_frames = sorted_frames / 4

        new_indeces,older_indeces = torch.sort(sorted_indeces)

        batch_size, time_frame ,hidden_dim = x.size()
        # gru output
        # layer gru
        out_hidden = self.getBiHidden(self.layer_gru,x,sorted_frames)
        # get a vector with fixed size (hidden_dim)
        sorted_frames = sorted_frames.view(-1,1)
        sorted_frames = sorted_frames.expand(batch_size,out_hidden.size(2))
        sorted_frames = sorted_frames.type(torch.cuda.FloatTensor)

        out_hidden = out_hidden.sum(dim=1)/sorted_frames

        x = out_hidden
        # target should be ordered
        out_bn = self.layer1(x)
        out_target = self.layer2(out_bn)
        # softmax
        predict_target = F.softmax(out_target, dim=1)

        # 计算loss
        tar_select_new = torch.gather(predict_target, 1, target)
        ce_loss = -torch.log(tar_select_new) 
        ce_loss = ce_loss.sum() / batch_size

        # 计算acc
        (data, predict) = predict_target.max(dim=1)
        prediction = predict[older_indeces]
        predict = predict.contiguous().view(-1,1)
        correct = predict.eq(target).float()       
        num_samples = predict.size(0)
        sum_acc = correct.sum().item()
        acc = sum_acc/num_samples

        return acc, ce_loss, prediction
class BiRes(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=2048, bn_dim=100, output_dim=10):
        super(BiRes, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bn_dim = bn_dim
        self.output_dim = output_dim

        #self.layer0 = nn.Sequential()
        #self.layer0.add_module('gru', nn.GRU(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False))
        self.layer1 = nn.Sequential()
        self.layer1.add_module('gru', nn.GRU(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('batchnorm', nn.BatchNorm1d(self.hidden_dim))
        self.layer2.add_module('linear', nn.Linear(self.hidden_dim, self.bn_dim))
        # self.layer2.add_module('Sigmoid', nn.Sigmoid())

        self.layer3 = nn.Sequential()
        self.layer3.add_module('batchnorm', nn.BatchNorm1d(self.bn_dim))
        self.layer3.add_module('linear', nn.Linear(self.bn_dim, self.output_dim))

    #def forward(self, src, mask, target):
    def forward(self, src, frames, target):
        batch_size, fea_frames, fea_dim = src.size()
        # squeeze frames:  [batch_size,1] --> [batch_size]
        frames = frames.squeeze()
        # get packed sequence
        sorted_frames,sorted_indeces = torch.sort(frames,descending=True)
        #print(sorted_frames)
        #print(sorted_frames.shape)
        #print(sorted_indeces.shape)
        # new input 
        src = src[sorted_indeces]
        src = pack_padded_sequence(src,sorted_frames.cpu().numpy(),batch_first=True)
        # new target
        target = target[sorted_indeces]


        # get gru output
        out_hidden, hidd = self.layer1(src)
        out_hidden,lengths = pad_packed_sequence(out_hidden,batch_first=True)

        # summation of the two hidden states in the same node
        # out_hidden = out_hidden[:,:,0:self.hidden_dim] + out_hidden[:,:,self.hidden_dim:]
        #mask = mask.contiguous().view(batch_size, fea_frames, 1).expand(batch_size, fea_frames, out_hidden.size(2))
        # output with new size (batch_size, hidden_dim)
        #out_hidden = out_hidden*mask
        # get a vector with fixed size length 
        sorted_frames = sorted_frames.view(-1,1)
        sorted_frames = sorted_frames.expand(batch_size,out_hidden.size(2))
        sorted_frames = sorted_frames.type(torch.cuda.FloatTensor)
        #print(sorted_frames)
        out_hidden = out_hidden.sum(dim=1)/sorted_frames

        out_hidden = out_hidden[:,0:self.hidden_dim] + out_hidden[:,self.hidden_dim:]
        # linear parts
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
        new_indeces , indeces = torch.sort(sorted_indeces)
        prediction = predict[indeces]
        #prediction = predict
        predict = predict.contiguous().view(-1,1)
        correct = predict.eq(target).float()       
        num_samples = predict.size(0)
        sum_acc = correct.sum().item()
        acc = sum_acc/num_samples

        return acc, ce_loss, prediction ,data
