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
import numpy as np
logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s[%(levelname)s] ---- %(message)s',
                    )

import torch
import torch.utils.data as Data

from mydata import TorchDataSet
from testmodel import inferModel as LanNet

## ======================================
# 配置文件和参数
# 数据列表
dev_list   = "../labels/label_dev_list_fb.txt"

# 保存模型地址
#model_dir = "./models"
#if not os.path.exists(model_dir):
#    os.makedirs(model_dir)
# cuda available or not
use_cuda = torch.cuda.is_available()
# 网络参数
dimension = 40
language_nums = 10
learning_rate = 0.1
batch_size = 1
chunk_num = 1
display_fre = 50
half = 4
fractional = 0.85

f=open("result.txt","w")
#f.write("posterior: changsha, hebei, nanchang, shanghai, kejia, minnan\n")

#fangyan=np.array(["hefei","nanchang","sichuan","changsha","shanghai","ningxia","minnan","kejia","hebei","shan3xi"])
fangyan=np.array(["minnan","nanchang","kejia","changsha","shanghai","hebei","hefei","shan3xi","sichuan","ningxia"])
#fangyan=np.array(["unknown","nanchang","unknown","unknown","unknown","unknown","unknown","unknown","unknown","unknown"])
sentences=[]

with open(dev_list,"r") as s:
    for line in s.readlines():
        sentences.append(line.strip().split("/")[-1].split()[0].replace("fb","pcm"))
sentences=np.array(sentences)
#print len(sentences)

## ======================================
dev_dataset = TorchDataSet(dev_list, batch_size, chunk_num, dimension)
logging.info('finish reading all train data')

train_module = LanNet(input_dim=dimension, hidden_dim=512, bn_dim=64, output_dim=language_nums)
logging.info(train_module)


if use_cuda:
    train_module = train_module.cuda()

train_module.load_state_dict(torch.load('models/infer8.model', map_location=lambda storage, loc: storage))
train_module.eval()
epoch_tic = time.time()
dev_loss = 0.
dev_acc = 0.
dev_batch_num = 0 
ACC = 0
dev_size = 0
start=0
size=0
for step, (batch_x, batch_y) in enumerate(dev_dataset): 
    #labels=batch_y.numpy()[:,0].astype(np.int32)
    #print batch_y
    print("step",step)
    size+=batch_y.size(0)
    sent=sentences[start:size]
    start=size
   # sent=sentences[batch_y.numpy()[:,1].astype(np.int32)]
    tic = time.time()
    bz = batch_y.size(0)
    dev_size += bz
    batch_target = batch_y[:,0].contiguous().view(-1, 1).long()
    batch_frames = batch_y[:,1].contiguous().view(-1, 1).long()

    max_batch_frames = int(max(batch_frames).item())
    batch_dev_data = batch_x[:, :max_batch_frames, :]

    step_batch_size = batch_target.size(0)

    if use_cuda:
        batch_dev_data   = batch_dev_data.cuda()
        batch_frames     = batch_frames.cuda()
        batch_target     = batch_target.cuda()

    with torch.no_grad():
        acc, loss,pre = train_module(batch_dev_data, batch_frames, batch_target)
   
    labels=fangyan[pre.cpu().numpy().astype(np.int32)]
    #print labels
    loss = loss.sum()/step_batch_size
    ACC += acc*bz
    toc = time.time()
    step_time = toc-tic

    dev_loss += loss.item()
    dev_acc += acc
    dev_batch_num += 1
    for s,label in zip(sent,labels):
        f.write(s+"\t"+label+"\n")

# 计算ACC是可选的，此处只是给选手提供一个计算ACC的例子，在最终的result中不需要输出ACC，选手在本地测试时可以输出ACC
#验证自己的代码。最终输出结果只需要按照官方文档中result.txt输出文件名和对应的方言种类即可，具体格式说明参考官方文档。
epoch_toc = time.time()
epoch_time = epoch_toc-epoch_tic
acc=dev_acc/dev_batch_num
ACC =ACC/dev_size
print("ACC:",ACC)
# f.write("ACC: "+str(ACC))
f.close()
