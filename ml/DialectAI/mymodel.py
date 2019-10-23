# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
#from warpctc_pytorch import CTCLoss
#from getPhonemes2 import dealMlf
import numpy as np


from resnet import resnet18

class pre_model(nn.Module):
    def __init__(self,hidden_dim=512):
        super(pre_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv = resnet18()
        self.layer1 = nn.Sequential()
        self.layer1.add_module('gru', nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True))
        self.layer2 = nn.Sequential()
        self.layer2.add_module('gru', nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True))
    def forward(self,x,frames,target):
        batch_size, fea_frames, fea_dim = x.size()
        # squeeze frames:  [batch_size,1] --> [batch_size]
        frames = frames.squeeze()
        # get packed sequence
        sorted_frames,sorted_indeces = torch.sort(frames,descending=True)
        # new input 
        x = x[sorted_indeces]
        # save the original x
        x_origin = x

        # conv output
        # new target
        target = target[sorted_indeces]

        x = x.unsqueeze(1)
        x = self.conv(x)

        # squeeze
        # B,F,T -> B,T,F
        x = x.squeeze()
        x = x.transpose(1,2)

        sorted_frames_origin  = sorted_frames
        sorted_frames         = sorted_frames / 4

        new_indeces,older_indeces = torch.sort(sorted_indeces)

        #print("out hidden shape",out_hidden.shape)
        return x_origin,x,target,older_indeces,sorted_frames_origin,sorted_frames


class LanNet(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=2048, bn_dim=100, output_dim=10):
        super(LanNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bn_dim = bn_dim
        self.output_dim = output_dim
        # phonemeSeq  dictionary

        self.layer_gru = nn.Sequential()
        self.layer_gru.add_module('gru', nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True))
        self.layer_gru_origin = nn.Sequential()
        self.layer_gru_origin.add_module('gru_origin', nn.GRU(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True))

        self.fractional = nn.Sequential()
        self.fractional.add_module('linear', nn.Linear(2, 1,bias=False))

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

    # two sorted frames

    # origin V.S. dealed after resnet
    # layer_gru_origin      V.S. layer_gru
    # x_origin              V.S. x
    # sorted_frames_origin  V.S. sorted_frames
    def forward(self, x_origin,x, sorted_frames_origin,sorted_frames,target):

        batch_size, time_frame ,hidden_dim = x.size()
        # gru output
        # layer gru
        out_hidden = self.getBiHidden(self.layer_gru,x,sorted_frames)
        # layer gru fb
        out_hidden_origin = self.getBiHidden(self.layer_gru_origin,x_origin,sorted_frames_origin)

        # get a vector with fixed size (hidden_dim)
        sorted_frames = sorted_frames.view(-1,1)
        sorted_frames = sorted_frames.expand(batch_size,out_hidden.size(2))
        sorted_frames = sorted_frames.type(torch.cuda.FloatTensor)

        out_hidden = out_hidden.sum(dim=1)/sorted_frames

        # get a vector with fixed size (hidden_dim) : fb
        sorted_frames_origin = sorted_frames_origin.view(-1,1)
        sorted_frames_origin = sorted_frames_origin.expand(batch_size,out_hidden_origin.size(2))
        sorted_frames_origin = sorted_frames_origin.type(torch.cuda.FloatTensor)

        out_hidden_origin = out_hidden_origin.sum(dim=1)/sorted_frames_origin



        # mixing out_hidden_origin and out_hidden with a proportion fractional
        #x = out_hidden_origin*(1.0 - fractional) + out_hidden*fractional
        # mixing out_hidden_origin and out_hidden with a neural network
        # (B,H) -> (B,H,1)
        out_hidden_origin = out_hidden_origin.unsqueeze(2)
        out_hidden        = out_hidden.unsqueeze(2)
        # (B,H,1) --> (B,H,2)
        x = torch.cat((out_hidden_origin,out_hidden),dim=2)

        # (B,H,2) --> (B*H,2)
        x = x.contiguous().view(-1,2)
        # (B*H,2) --> (B*H,1)
        x = self.fractional(x) 
        # (B*H,1) --> (B*H)
        x = x.squeeze()
        # (B*H) --> (B,H)
        x = x.contiguous().view(batch_size,-1)

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
        predict = predict.contiguous().view(-1,1)
        correct = predict.eq(target).float()       
        num_samples = predict.size(0)
        sum_acc = correct.sum().item()
        acc = sum_acc/num_samples

        return acc, ce_loss
