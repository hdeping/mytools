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

        #self.layer0 = nn.Sequential()
        #self.layer0.add_module('gru', nn.GRU(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False))
        self.layer1 = nn.Sequential()
        self.layer1.add_module('lstm', nn.LSTM(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('batchnorm', nn.BatchNorm1d(self.hidden_dim))
        self.layer2.add_module('linear', nn.Linear(self.hidden_dim, self.bn_dim))
        # self.layer2.add_module('Sigmoid', nn.Sigmoid())

        self.layer3 = nn.Sequential()
        self.layer3.add_module('batchnorm', nn.BatchNorm1d(self.bn_dim))
        self.layer3.add_module('linear', nn.Linear(self.bn_dim, self.output_dim))

    def forward(self, src):
        batch_size, fea_frames, fea_dim = src.size()

        out_hidden, hidd = self.layer1(src)
        #print(out_hidden.data.shape)
        out_hidden = out_hidden.contiguous().view(-1, out_hidden.size(-1))   
        #print(out_hidden.data.shape)
        out_bn = self.layer2(out_hidden)
        out_target = self.layer3(out_bn)

        return out_target
