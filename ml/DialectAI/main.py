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
from read_data import  TorchDataSet
from mymodel import LanNet

## ======================================
# data list
# train
train_list = "../labels/label_train_list_fb.txt"
# dev
dev_list   = "../labels/label_dev_list_fb.txt"

# basic configuration parameter
use_cuda = torch.cuda.is_available()
# network parameter 
dimension = 40 # 40 before
language_nums = 10 # 9!
learning_rate = 0.1
batch_size = 64
chunk_num = 10
#train_iteration = 10
train_iteration = 12
display_fre = 50
half = 4
# data augmentation

# save the models
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

## ======================================
# with data augmentation
train_dataset = TorchDataSet(train_list, batch_size, chunk_num, dimension)
# without data augmentation
dev_dataset = TorchDataSet(dev_list, batch_size, chunk_num, dimension)
logging.info('finish reading all train data')

# 优化器，SGD更新梯度
train_module = LanNet(input_dim=dimension, hidden_dim=128, bn_dim=30, output_dim=language_nums)
logging.info(train_module)
optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)

# initialize the model
#train_module.load_state_dict(torch.load("models/model9.model"))
#device = torch.device("cuda:2")
# 将模型放入GPU中
if use_cuda:
    # torch 0.4.0
    #train_module = train_module.to(device)
    # torch 0.3.0
    train_module = train_module.cuda()

# regularization factor
factor = 0.0005

for epoch in range(0,train_iteration):
    print("epoch",epoch)
    if epoch == 4:
        learning_rate = 0.05
        optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)
    if epoch == 8:
        learning_rate = 0.02
        optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)
##  train
    train_dataset.reset()
    train_module.train()
    epoch_tic = time.time()
    train_loss = 0.
    train_acc = 0.

    sum_batch_size = 0
    curr_batch_size = 0
    curr_batch_acc = 0
    tic = time.time()
    for step, (batch_x, batch_y) in enumerate(train_dataset): 
        #print("step is ",step)
        batch_target = batch_y[:,0].contiguous().view(-1, 1).long()
        batch_frames = batch_y[:,1].contiguous().view(-1, 1)

        #max_batch_frames = int(max(batch_frames).item())
        #print(dir(batch_frames))
        max_batch_frames = int(max(batch_frames).item())
        #print(batch_x.data.shape)
        batch_train_data = batch_x[:, :max_batch_frames, :]
        #print(batch_train_data.data.shape)

        step_batch_size = batch_target.size(0)
        batch_mask = torch.zeros(step_batch_size, max_batch_frames)
        for ii in range(step_batch_size):
            frames = int(batch_frames[ii].item())
            batch_mask[ii, :frames] = 1.

        # 将数据放入GPU中
        if use_cuda:
            # torch 0.4.0
            #batch_train_data = batch_train_data.to(device)
            #batch_mask       = batch_mask.to(device)
            #batch_target     = batch_target.to(device)
            # torch 0.3.0
            batch_train_data = batch_train_data.cuda()
            batch_mask       = batch_mask.cuda()
            batch_target     = batch_target.cuda()

        acc, loss = train_module(batch_train_data, batch_mask, batch_target)
        
        # loss = loss.sum()
        backward_loss = loss
        optimizer.zero_grad()
        # L1 regularization 
        #l1_crit = torch.nn.L1Loss(size_average=False)
        #l1_crit.cuda()
        # L4 regularization
        reg_loss = 0
        for param in train_module.parameters():
            #reg_loss += l1_crit(param)
            #reg_loss += param.norm(2)
            reg_loss += param.norm(4)**4
        backward_loss += factor * reg_loss
                
        # get the gradients
        backward_loss.backward()
        # update the weights
        optimizer.step()


        train_loss += loss.item()
        train_acc += acc
        curr_batch_acc += acc
        sum_batch_size += 1
        curr_batch_size += 1
        if step % display_fre == 0:
            toc = time.time()
            step_time = toc-tic
            logging.info('Epoch:%d, Batch:%d, acc:%.6f, loss:%.6f, cost time :%.6fs', epoch, step, curr_batch_acc/curr_batch_size, loss.item(), step_time)
            curr_batch_acc = 0.
            curr_batch_size = 0
            tic = toc


    
    modelfile = '%s/model%d.model'%(model_dir, epoch)
    torch.save(train_module.state_dict(), modelfile)
    epoch_toc = time.time()
    epoch_time = epoch_toc-epoch_tic
    logging.info('Epoch:%d, train-acc:%.6f, train-loss:%.6f, cost time :%.6fs', epoch, train_acc/sum_batch_size, train_loss/sum_batch_size, epoch_time)

##  -----------------------------------------------------------------------------------------------------------------------------
##  dev
    train_module.eval()
    epoch_tic = time.time()
    dev_loss = 0.
    dev_acc = 0.
    dev_batch_num = 0 

    for step, (batch_x, batch_y) in enumerate(dev_dataset): 
        tic = time.time()

        batch_target = batch_y[:,0].contiguous().view(-1, 1).long()
        batch_frames = batch_y[:,1].contiguous().view(-1, 1)

        max_batch_frames = int(max(batch_frames).item())
        batch_dev_data = batch_x[:, :max_batch_frames, :]

        step_batch_size = batch_target.size(0)
        batch_mask = torch.zeros(step_batch_size, max_batch_frames)
        for ii in range(step_batch_size):
            frames = int(batch_frames[ii].item())
            batch_mask[ii, :frames] = 1.

        # 将数据放入GPU中
        if use_cuda:
            # torch 0.4.0
            #batch_dev_data   = batch_dev_data.to(device)
            #batch_mask       = batch_mask.to(device)
            #batch_target     = batch_target.to(device)
            # torch 0.3.0
            batch_dev_data   = batch_dev_data.cuda()
            batch_mask       = batch_mask.cuda()
            batch_target     = batch_target.cuda()
            
        with torch.no_grad():
            #acc, loss = train_module(batch_dev_data, batch_mask, batch_target)
            acc, loss = train_module(batch_dev_data, batch_mask, batch_target)
        
        loss = loss.sum()/step_batch_size

        toc = time.time()
        step_time = toc-tic

        dev_loss += loss.item()
        dev_acc += acc
        dev_batch_num += 1
    
    epoch_toc = time.time()
    epoch_time = epoch_toc-epoch_tic
    acc=dev_acc/dev_batch_num
    logging.info('Epoch:%d, dev-acc:%.6f, dev-loss:%.6f, cost time :%.6fs', epoch, acc, dev_loss/dev_batch_num, epoch_time)
