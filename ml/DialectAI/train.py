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
import numpy as np

#from mydata import get_samples, get_data, TorchDataSet
from mydata import  TorchDataSet
from mymodel import LanNet

## ======================================
# data list
# train
train_list = "label_train_list_fb.txt"
# dev
dev_list   = "label_dev_list_fb.txt"

mlf_file  = "../label/train.dev"

# basic configuration parameter
use_cuda = torch.cuda.is_available()
#use_cuda = False
# network parameter 
dimension = 40 # 40 before
language_nums = 10  # 9!
learning_rate = 1e-4
batch_size = 8
chunk_num = 10
#train_iteration = 10
train_iteration = 12
display_fre = 20
half = 4
# data augmentation

# save the models
import sys
#model_dir = "models" + sys.argv[1]
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# seed 
torch.manual_seed(time.time())

## ======================================
# with data augmentation
train_dataset = TorchDataSet(train_list, batch_size, chunk_num, dimension)
# without data augmentation
dev_dataset = TorchDataSet(dev_list, batch_size, chunk_num, dimension)
logging.info('finish reading all train data')

# 优化器，SGD更新梯度

# initialize the model
#train_module.load_state_dict(torch.load("models/model9.model"))
#device = torch.device("cuda:2")

def train(count):    
    # 将模型放入GPU中
    train_module = LanNet(input_dim=dimension, hidden_dim=128, bn_dim=30, output_dim=language_nums)
    if count == 0:
        logging.info(train_module)
    if use_cuda:
        # torch 0.4.0
        #train_module = train_module.to(device)
        # torch 0.3.0
        train_module = train_module.cuda()
    
    # regularization factor
    factor = 0.0005
    optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)
    for epoch in range(0,train_iteration):
        print("epoch",epoch)
        #if epoch == 4:
        #    learning_rate = 0.05
        #    optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)
        #if epoch == 8:
        #    learning_rate = 0.02
        #    optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)
    ##  train
        train_dataset.reset()
        train_module.train()
        epoch_tic = time.time()
        train_loss = 0.
    
        sum_batch_size = 0
        curr_batch_size = 0
        tic = time.time()
        for step, (batch_x, batch_y,name_list) in enumerate(train_dataset): 
            #print("step is ",step)
            batch_target = batch_y[:,0].contiguous().view(-1, 1).long()
            batch_frames = batch_y[:,1].contiguous().view(-1, 1).long()
            #print(batch_frames)
            #print(len(name_list),batch_x.shape,batch_target.shape)
            #print(np.array(name_list))
            name_list = np.array(name_list)
            #print(len(name_list))
    
            #max_batch_frames = int(max(batch_frames).item())
            #print(dir(batch_frames))
            max_batch_frames = int(max(batch_frames).item())
            #print(batch_x.data.shape)
            batch_train_data = batch_x[:, :max_batch_frames, :]
            #print(batch_train_data.data.shape)
    
            step_batch_size = batch_target.size(0)
            #batch_mask = torch.zeros(step_batch_size, max_batch_frames)
            #for ii in range(step_batch_size):
            #    frames = int(batch_frames[ii].item())
            #    batch_mask[ii, :frames] = 1.
    
            # 将数据放入GPU中
            if use_cuda:
                # torch 0.4.0
                #batch_train_data = batch_train_data.to(device)
                #batch_mask       = batch_mask.to(device)
                #batch_target     = batch_target.to(device)
                # torch 0.3.0
                batch_train_data = batch_train_data.cuda()
                #batch_mask       = batch_mask.cuda()
                batch_frames       = batch_frames.cuda()
                batch_target     = batch_target.cuda()
    
            #acc, loss = train_module(batch_train_data, batch_frames, batch_target)
            loss = train_module(batch_train_data, batch_frames, name_list)
            
            # loss = loss.sum()
            backward_loss = loss
            optimizer.zero_grad()
            # L1 regularization 
            #l1_crit = torch.nn.L1Loss(size_average=False)
            #l1_crit.cuda()
            #reg_loss = 0
            #for param in train_module.parameters():
            #    #reg_loss += l1_crit(param)
            #    reg_loss += param.norm(2)
            #backward_loss += factor * reg_loss
                    
            # get the gradients
            backward_loss.backward()
            # update the weights
            optimizer.step()
    
    
            train_loss += loss.item()
            sum_batch_size += 1
            curr_batch_size += 1

            if step % display_fre == 0:
                toc = time.time()
                step_time = toc-tic
                logging.info('Epoch:%d, Batch:%d,  loss:%.6f, cost time :%.6fs', epoch, step, loss.item(), step_time)
                curr_batch_acc = 0.
                curr_batch_size = 0
                tic = toc
    
    
        
        epoch_toc = time.time()
        epoch_time = epoch_toc-epoch_tic
        logging.info('Epoch:%d, train-loss:%.6f, cost time :%.6fs', epoch, train_loss/sum_batch_size, epoch_time)
        modelfile = '%s/model%d-%d.model'%(model_dir,epoch,count)
        torch.save(train_module.state_dict(), modelfile)

for count in range(1):
    train(count)
