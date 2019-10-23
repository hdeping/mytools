# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from warpctc_pytorch import CTCLoss
from getPhonemes2 import dealMlf
import numpy as np


from resnet import resnet18
from tcn import TemporalConvNet


class LanNet(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=2048, bn_dim=100, output_dim=10):
        super(LanNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bn_dim = bn_dim
        self.output_dim = output_dim
        # phonemeSeq  dictionary
        self.phonemes_dict = dealMlf("../labels/train.mlf")

        #self.conv  = resnet18()
        self.conv = TemporalConvNet(self.hidden_dim,[self.hidden_dim,160,256,320,512])

        self.layer1 = nn.Sequential()
        self.layer1.add_module('gru', nn.GRU(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True))
        self.layer2 = nn.Sequential()
        self.layer2.add_module('gru', nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True))
        self.layer3 = nn.Sequential()
        self.layer3.add_module('gru', nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True))
        #self.layer4 = nn.Sequential()
        #self.layer4.add_module('gru', nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True))
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


    # get phoneme sequence
    def phonemeSeq(self,name_list):
        labels_sizes = []
        extension = '.fb'
        # the first sample
        name = name_list[0] + extension
        labels = self.phonemes_dict[name]
        labels_sizes.append(len(labels))
    
        # the other ones
        for name in name_list[1:]:
            name = name + extension
            arr  = self.phonemes_dict[name]
            labels = np.concatenate((labels,arr))
            labels_sizes.append(len(arr))
    
        #labels_sizes = np.array(labels_sizes)
        return labels,labels_sizes

    def forward(self, src, frames,name_list):
        #print(src.shape)
        batch_size, fea_frames, fea_dim = src.size()
        # squeeze frames:  [batch_size,1] --> [batch_size]
        frames = frames.squeeze()
        # get packed sequence
        sorted_frames,sorted_indeces = torch.sort(frames,descending=True)

        # new input 
        src = src[sorted_indeces]
        # new name_list
        #print(sorted_indeces)
        #print("name list length",len(name_list))
        name_list = name_list[sorted_indeces]

        #src = pack_padded_sequence(src,sorted_frames.cpu().numpy(),batch_first=True)
        # new target
        #target = target[sorted_indeces]
        #print(sorted_frames)
        #print(name_list)



        # get gru output
        # layer 1
        #sorted_frames = sorted_frames / 4
        out_hidden = self.getBiHidden(self.layer1,src,sorted_frames)

        out_hidden = self.getBiHidden(self.layer2,out_hidden,sorted_frames)
        # layer2
        out_hidden_new = self.getBiHidden(self.layer3,out_hidden,sorted_frames)
        ## layer3
        #out_hidden_new = self.getBiHidden(self.layer3,out_hidden_new,sorted_frames)

        # residual part
        out_hidden = out_hidden + out_hidden_new

        # conv output

        #print(out_hidden.shape)


        #out_hidden = out_hidden.unsqueeze(1)

        # out_hidden : B,T,F
        out_hidden = out_hidden.transpose(1,2)
        out_hidden = self.conv(out_hidden)
        #print(out_hidden.shape)

        # transpose
        # out_hidden : B,F,T -> B,T,F
        out_hidden = out_hidden.transpose(1,2)
        # out_hidden : B,T,F -> T,B,F
        out_hidden = out_hidden.transpose(0,1)

        #print(out_hidden.shape)
        # get labels and labels_sizes
        labels, labels_sizes = self.phonemeSeq(name_list)
        # tensor to torch (cuda))
        #labels       = torch.cuda.IntTensor(labels)
        #labels_sizes = torch.cuda.IntTensor(labels_sizes)
        labels       = torch.IntTensor(labels)
        labels_sizes = torch.IntTensor(labels_sizes)

        # CTC 
        ctc_loss = CTCLoss()
        frames = sorted_frames.cpu().type(torch.IntTensor)
        probs = out_hidden.cpu().type(torch.FloatTensor)
        loss = ctc_loss(probs, labels, frames, labels_sizes)

        loss = loss.cuda()
        print(loss)
        return loss
