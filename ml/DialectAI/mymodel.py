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
        self.hidden_node = self.hidden_dim // 4
        self.input_node = self.input_dim// 4
        self.layer0 = nn.Sequential()
        self.layer0.add_module('gru', nn.GRU(self.input_node, self.hidden_node, num_layers=1, batch_first=True, bidirectional=False))
        self.layer1 = nn.Sequential()
        self.layer1.add_module('gru', nn.GRU(self.input_node, self.hidden_node, num_layers=1, batch_first=True, bidirectional=False))
        self.layerA = nn.Sequential()
        self.layerA.add_module('gru', nn.GRU(self.input_node, self.hidden_node, num_layers=1, batch_first=True, bidirectional=False))
        self.layerB = nn.Sequential()
        self.layerB.add_module('gru', nn.GRU(self.input_node, self.hidden_node, num_layers=1, batch_first=True, bidirectional=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('batchnorm', nn.BatchNorm1d(self.hidden_dim))
        self.layer2.add_module('linear', nn.Linear(self.hidden_dim, self.bn_dim))
        # self.layer2.add_module('Sigmoid', nn.Sigmoid())

        self.layer3 = nn.Sequential()
        self.layer3.add_module('batchnorm', nn.BatchNorm1d(self.bn_dim))
        self.layer3.add_module('linear', nn.Linear(self.bn_dim, self.output_dim))

    def forward(self, src, mask, target):
        batch_size, fea_frames, fea_dim = src.size()

        # get gru output

        node = [i*self.input_node for i in range(5)]
        #print(node)
        out_hidden = []
        layer = [self.layer0,self.layer1,self.layerA,self.layerB]
        for i in range(4):
            out_hidden0, hidd = layer[i](src[:,:,node[i]:node[i+1]])
            out_hidden.append(out_hidden0)
        # combine the four parts
        out_hidden = torch.cat(tuple(out_hidden),dim=2)
        #print(out_hidden.shape)
        #print(out_hidden.shape)
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
        predict = predict.contiguous().view(-1,1)
        correct = predict.eq(target).float()       
        num_samples = predict.size(0)
        sum_acc = correct.sum().item()
        acc = sum_acc/num_samples

        return acc, ce_loss
