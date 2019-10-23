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
        self.layer1.add_module('gru', nn.GRU(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('batchnorm', nn.BatchNorm1d(self.hidden_dim))
        self.layer2.add_module('linear', nn.Linear(self.hidden_dim, self.bn_dim))
        # self.layer2.add_module('Sigmoid', nn.Sigmoid())

        self.layer3 = nn.Sequential()
        self.layer3.add_module('batchnorm', nn.BatchNorm1d(self.bn_dim))
        self.layer3.add_module('linear', nn.Linear(self.bn_dim, self.output_dim))

    def forward(self, src,mask,target):
        batch_size, fea_frames, fea_dim = src.size()

        out_hidden, hidd = self.layer1(src)
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

class baseConv1d(nn.Module):
    def __init__(self,input_chanel,output_chanel,kernel_size,stride,padding):
        super(baseConv1d,self).__init__()
        # architeture of the base conv1d
        self.conv = nn.Conv1d(input_chanel,output_chanel,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn   = nn.BatchNorm1d(output_chanel)
    def forward(self,x):
        # conv 
        #print(x.size())
        x = self.conv(x)
        # batchnorm 
        x = self.bn(x)
        # 1d avg pool 
        x = F.max_pool1d(x,kernel_size=2)
        # relu output
        x = F.relu(x)
        return x
class baseLinear(nn.Module):
    def __init__(self,input_chanel,output_chanel):
        super(baseLinear,self).__init__()

        self.fc = nn.Linear(input_chanel,output_chanel)
    def forward(self,x):
        x = self.fc(x)
        x = F.relu(x)
        return x


class featureModel(nn.Module):
    def __init__(self, input_dim=48):
        super(featureModel, self).__init__()
        self.input_dim = input_dim
        
        self.layer = nn.Sequential()
        self.layer.add_module('fc1', baseLinear(self.input_dim,1024))
        self.layer.add_module('fc2', baseLinear(1024,1024))
        self.layer.add_module('fc3', baseLinear(1024,40))

    def forward(self, x):
        batch_size, fea_frames, fea_dim = x.size()
        #print(x.size)
        # reshape the input
        x = x.contiguous().view(batch_size*fea_frames,-1)
        out_target = self.layer(x)
        out_target = out_target.contiguous().view(batch_size,fea_frames,-1)

        return out_target

class featureModel_old(nn.Module):
    def __init__(self, input_dim=48):
        super(featureModel_old, self).__init__()
        self.input_dim = input_dim
        
        self.layer = nn.Sequential()
        self.layer.add_module('fc1', baseLinear(self.input_dim,1024))
        self.layer.add_module('fc2', baseLinear(1024,1024))
        self.layer.add_module('fc3', baseLinear(1024,40))
        self.layer.add_module('fc4', baseLinear(40,1024))
        self.layer.add_module('fc5', baseLinear(1024,1024))
        self.layer.add_module('fc6', baseLinear(1024,self.input_dim))

    def forward(self, x):
        batch_size, fea_frames, fea_dim = x.size()
        #print(x.size)
        # reshape the input
        x = x.contiguous().view(batch_size*fea_frames,-1)
        out_target = self.layer(x)

        return out_target
def getModel(dimension,language_nums):
    train_module = featureModel(input_dim=dimension)
    model = featureModel_old(input_dim=dimension)
    model = nn.DataParallel(model,device_ids=[0,1])
    #logging.info(model)
    model.load_state_dict(torch.load('./models/model19.model'))
    #old_keys = model.state_dict().keys()
    old_keys = model.state_dict().keys()
    keys = []
    # convert to the subscripted array
    #for key in train_module.state_dict().keys():
    #    print(key)
    for key in old_keys:
        #print(key)
        keys.append(key)
    #print(keys)
    # new dict
    dict_new = train_module.state_dict().copy()
    for key in keys[:6]:
        print(key)
        key1 = key.replace('module.','')
        dict_new[key1] =  model.state_dict()[key]
    # load parameter
    train_module.load_state_dict(dict_new)
    return train_module
#train_module = getModel(320,10)
