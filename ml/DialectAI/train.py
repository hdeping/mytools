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
from mymodel import LanNet

## ======================================
# data list
# train
train_list = "label_train_list_fb.txt"
train_mlf  = "train.mlf"
#train_list = "all.txt"
#train_mlf  = "all.mlf"
# dev
dev_list   = "label_dev_list_fb.txt"
dev_mlf    = "dev.mlf"

# basic configuration parameter
use_cuda = torch.cuda.is_available()
# network parameter 
dimension = 8 # 40 before
dimension2 = 13 # 40 before
language_nums = 10 # 9!
learning_rate = 0.2
batch_size = 64
chunk_num = 10
#train_iteration = 10
train_iteration = 60
display_fre = 50
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
train_dataset = TorchDataSet(train_list, batch_size, chunk_num, dimension, dimension2, train_mlf)
# without data augmentation
dev_dataset = TorchDataSet(dev_list, batch_size, chunk_num, dimension, dimension2, dev_mlf)
logging.info('finish reading all train data')

# 优化器，SGD更新梯度
train_module = LanNet(input_dim_fb=dimension2,
                      input_dim_phoneme = dimension, 
                      hidden_dim=128,  
                      bn_dim=30,  
                      output_dim=language_nums)

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
def getLR(epoch):
    if epoch == 15:
        learning_rate = 0.1
        optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)
    if epoch == 30:
        learning_rate = 0.02
        optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)
    if epoch == 45:
        learning_rate = 0.001
        optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)
    #if epoch == 40:
    #    learning_rate = 0.003
    #    optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)
    #if epoch == 80:
    #    learning_rate = 0.001
    #    optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)
    #if epoch == 120:
    #    learning_rate = 0.0001
    #    optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)

def getBatchData(batch_fb,batch_phoneme, batch_y):

    batch_target = batch_y[:,0].contiguous().view(-1, 1).long()
    batch_frames_fb = batch_y[:,1].contiguous().view(-1, 1)
    batch_frames_phoneme= batch_y[:,2].contiguous().view(-1, 1)
    #print(batch_target.shape,batch_frames_fb.shape,batch_frames_phoneme.shape)
    
    # fb
    max_batch_frames_fb = int(max(batch_frames_fb).item())
    batch_train_data_fb = batch_fb[:, :max_batch_frames_fb, :]
    # phoneme
    max_batch_frames_phoneme = int(max(batch_frames_phoneme).item())
    batch_train_data_phoneme = batch_phoneme[:, :max_batch_frames_phoneme, :]
    #print(max_batch_frames_fb,max_batch_frames_phoneme)
    #print(batch_frames_fb)
    #print(batch_frames_phoneme)
    
    step_batch_size = batch_target.size(0)
    # fb mask
    batch_mask_fb = torch.zeros(step_batch_size, max_batch_frames_fb)
    # phoneme mask
    batch_mask_phoneme = torch.zeros(step_batch_size, max_batch_frames_phoneme)
    for ii in range(step_batch_size):
        frames = int(batch_frames_fb[ii].item())
        batch_mask_fb[ii, :frames] = 1.
        frames = int(batch_frames_phoneme[ii].item())
        batch_mask_phoneme[ii, :frames] = 1.
    
    # 将数据放入GPU中
    if use_cuda:
        # torch 0.4.0
        #batch_train_data = batch_train_data.to(device)
        #batch_mask       = batch_mask.to(device)
        #batch_target     = batch_target.to(device)
        # torch 0.3.0
        # fb
        batch_train_data_fb = batch_train_data_fb.cuda()
        batch_mask_fb       = batch_mask_fb.cuda()
        # phoneme
        batch_train_data_phoneme = batch_train_data_phoneme.cuda()
        batch_mask_phoneme       = batch_mask_phoneme.cuda()
    
        batch_target     = batch_target.cuda()
    # return data
    return batch_train_data_fb,batch_mask_fb,batch_train_data_phoneme,batch_mask_phoneme,batch_target

for epoch in range(0,train_iteration):
    print("epoch",epoch)
    getLR(epoch)
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
    for step, (batch_fb,batch_phoneme, batch_y) in enumerate(train_dataset): 
        #print("step is ",step)

        batch_train_data_fb,batch_mask_fb,  \
        batch_train_data_phoneme,batch_mask_phoneme, \
        batch_target = getBatchData(batch_fb,batch_phoneme, batch_y)

        acc, loss = train_module(batch_train_data_fb,batch_train_data_phoneme, \
                                 batch_mask_fb,batch_mask_phoneme, batch_target)
        
        # loss = loss.sum()
        backward_loss = loss
        optimizer.zero_grad()
        # L1 regularization 
        #l1_crit = torch.nn.L1Loss(size_average=False)
        #l1_crit.cuda()
        reg_loss = 0
        for param in train_module.parameters():
            #reg_loss += l1_crit(param)
            reg_loss += param.norm(2)
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

    for step, (batch_fb,batch_phoneme, batch_y) in enumerate(dev_dataset): 
        tic = time.time()

        # get batch data
        batch_train_data_fb,batch_mask_fb,  \
        batch_train_data_phoneme,batch_mask_phoneme, \
        batch_target = getBatchData(batch_fb,batch_phoneme, batch_y)


            
        with torch.no_grad():
            acc, loss = train_module(batch_train_data_fb,batch_train_data_phoneme, \
                                     batch_mask_fb,batch_mask_phoneme, batch_target)
        
        step_batch_size = batch_target.size(0)
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
