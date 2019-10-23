# -*- coding:utf-8 -*-

# --------------------------------------------------------- #
#                                                           #
#                Train Language Recognition                 #
#                                                           #
# --------------------------------------------------------- #
#                                                           #
#                  Train Language Recognition               #
#          Copyright(c) iFlytek Corporation, 2018           #
#                    Hefei, Anhui, PRC                      #
#                   http://www.iflytek.com                  #
#                                                           #
# --------------------------------------------------------- #
#  python  : 2.7 version                                    #
#  cuda    : Toolkit 9.1                                    #
#  pytorch : 0.4.0                                          #
# --------------------------------------------------------- #

import os
import time
import codecs
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s[%(levelname)s] ---- %(message)s',
                    )

import torch
import torch.utils.data as Data

#from read_data import get_samples, get_data, TorchDataSet
from mydata import  TorchDataSet
from testmodel import LanNet,BiRes

## ======================================
# data list
# train
dev_list   = "../labels/label_dev_list_fb.txt"

# basic configuration parameter
use_cuda = torch.cuda.is_available()
# network parameter 
toneLengthD = 6
dimension = 40
language_nums = 10 # 9!
learning_rate = 0.1
batch_size = 50
chunk_num = 10
#train_iteration = 10
train_iteration = 16
display_fre = 50
half = 4

hidden_dim = 512
bn_dim     = 64


## ======================================
# with data augmentation
# without data augmentation
dev_dataset = TorchDataSet(dev_list, batch_size, chunk_num, dimension)
logging.info('finish reading all train data')



def test1(train_module):
    train_module.eval()
    epoch_tic = time.time()
    dev_loss = 0.
    dev_acc = 0.
    dev_batch_num = 0 
    
    result_target = []
    for step, (batch_x, batch_y) in enumerate(dev_dataset): 
        print("step is ",step)
        tic = time.time()
    
        batch_target = batch_y[:,0].contiguous().view(-1, 1).long()
        batch_frames = batch_y[:,1].contiguous().view(-1, 1).long()
    
        max_batch_frames = int(max(batch_frames).item())
        batch_dev_data = batch_x[:, :max_batch_frames, :]
    
        step_batch_size = batch_target.size(0)
    
        # 将数据放入GPU中
        if use_cuda:
            batch_dev_data   = batch_dev_data.cuda()
            batch_frames       = batch_frames.cuda()
            batch_target     = batch_target.cuda()
            
        with torch.no_grad():
            acc, loss,prediction = train_module(batch_dev_data, batch_frames, batch_target)
        for i in range(batch_size):
            result_target.append([batch_target[i].item(),prediction[i].item()])
            #result_target.append(prediction[i].item())
        
        loss = loss.sum()/step_batch_size
    
        toc = time.time()
        step_time = toc-tic
    
        dev_loss += loss.item()
        dev_acc += acc
        dev_batch_num += 1
    
    epoch_toc = time.time()
    epoch_time = epoch_toc-epoch_tic
    acc=dev_acc/dev_batch_num
    logging.info('dev-acc:%.6f, dev-loss:%.6f, cost time :%.6fs', acc, dev_loss/dev_batch_num, epoch_time)
    return result_target
def test2(train_module):
    train_module.eval()
    epoch_tic = time.time()
    dev_loss = 0.
    dev_acc = 0.
    dev_batch_num = 0 
    
    result_target = []
    for step, (batch_x, batch_y) in enumerate(dev_dataset): 
        print("step is ",step)
        tic = time.time()
    
        batch_target = batch_y[:,0].contiguous().view(-1, 1).long()
        batch_frames = batch_y[:,1].contiguous().view(-1, 1).long()
    
        max_batch_frames = int(max(batch_frames).item())
        batch_dev_data = batch_x[:, :max_batch_frames, :]
    
        step_batch_size = batch_target.size(0)
    
        # 将数据放入GPU中
        if use_cuda:
            batch_dev_data   = batch_dev_data.cuda()
            batch_frames       = batch_frames.cuda()
            batch_target     = batch_target.cuda()
            
        with torch.no_grad():
            acc, loss,prediction,data = train_module(batch_dev_data, batch_frames, batch_target)
        for i in range(batch_size):
            result_target.append([prediction[i].item(),data[i].item()])
        
        loss = loss.sum()/step_batch_size
    
        toc = time.time()
        step_time = toc-tic
    
        dev_loss += loss.item()
        dev_acc += acc
        dev_batch_num += 1
    
    epoch_toc = time.time()
    epoch_time = epoch_toc-epoch_tic
    acc=dev_acc/dev_batch_num
    logging.info('dev-acc:%.6f, dev-loss:%.6f, cost time :%.6fs', acc, dev_loss/dev_batch_num, epoch_time)
    return result_target
# output the result
# res_target: (N,2)   label    and predicted label
# target_prop: (N,2)  predicted and probability
def getNewTarget(res_target,target_prop):
    # select the language with the label 3,6,8
    # changsha,hebei,sichuan
    res_target = torch.IntTensor(res_target)
    select = torch.randn(len(res_target))
    for i,line in enumerate(target_prop):
        if line[0] == 3 or line[0] == 6 or line[0] == 8:
            select[i] = line[1]
        else:
            select[i] = 0
    # sort the select in a descending order
    
    data,sorted_indeces = torch.sort(select,descending=True)

    #print(data[:10],sorted_indeces[:10])
    # select previous 1500 ones
    sorted_indeces = sorted_indeces[:1500]
    for order in sorted_indeces:
        # replace the target
        print(order)
        res_target[order,1] = target_prop[order][0]

    # return res_target
    return res_target


import numpy as np
# model 1
train_module = LanNet(input_dim=dimension, hidden_dim=hidden_dim, bn_dim=bn_dim, output_dim=language_nums)
logging.info(train_module)

# 将模型放入GPU中
if use_cuda:
    train_module = train_module.cuda()
train_module.load_state_dict(torch.load("infer/infer.model"))
result_target1 = test1(train_module)

# model 2
train_module = BiRes(input_dim=dimension, hidden_dim=128, bn_dim=30, output_dim=language_nums)
logging.info(train_module)

# 将模型放入GPU中
if use_cuda:
    train_module = train_module.cuda()
train_module.load_state_dict(torch.load("infer/bi-res.model"))
result_target2 = test2(train_module)


target = getNewTarget(result_target1,result_target2)
correct = 0
for line in target:
    if line[0] == line[1]:
        correct = correct + 1

print("correct = ",correct)

