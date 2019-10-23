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
from testmodel import LanNet

## ======================================
# data list
# train
dev_list   = "label_dev_list_fb.txt"

# basic configuration parameter
use_cuda = torch.cuda.is_available()
# network parameter 
toneLengthD = 6
dimension = 40
data_dimension = 320
language_nums = 10 # 9!
learning_rate = 0.1
batch_size = 50
chunk_num = 10
#train_iteration = 10
train_iteration = 16
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
# without data augmentation
dev_dataset = TorchDataSet(dev_list, batch_size, chunk_num, dimension)
logging.info('finish reading all train data')

# 优化器，SGD更新梯度
train_module = LanNet(input_dim=dimension, hidden_dim=256, bn_dim=30, output_dim=language_nums)
logging.info(train_module)
optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)

# initialize the model
#device = torch.device("cuda:2")
# 将模型放入GPU中
if use_cuda:
    # torch 0.4.0
    #train_module = train_module.to(device)
    # torch 0.3.0
    train_module = train_module.cuda()

# regularization factor
factor = 0.0005

##  -----------------------------------------------------------------------------------------------------------------------------
##  dev
def test():
    train_module.eval()
    epoch_tic = time.time()
    dev_loss = 0.
    dev_acc = 0.
    dev_batch_num = 0 
    
    result_target = []
    for step, (batch_x, batch_y) in enumerate(dev_dataset): 
        #print("step is ",step)
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
            acc, loss,prediction = train_module(batch_dev_data, batch_mask, batch_target)
        #print(batch_target,prediction)
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
# output the result
import numpy as np
result = []
for i in range(6):
    print("model ",i)
    print("loading model9-%d.model"%(i))
    train_module.load_state_dict(torch.load("models/model9-%d.model"%(i)))
    result_target = test()
    result.append(result_target)

# deal with the output
result = np.array(result)
result = np.transpose(result,(1,0,2))
print(result.shape)
size = len(result)
#new = np.zeros((5000,2*size))
#new[:,0] = result[:,0,0]
#new[:,1:] = result[:,:,1]
result = np.reshape(result,(size,-1))
print(result.shape)
np.savetxt("result.txt",result,fmt='%d')
