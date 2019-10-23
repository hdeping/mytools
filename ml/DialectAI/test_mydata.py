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

from mydata import  TorchDataSet
from mymodel import LanNet
from getPhonemes2 import dealMlf

## ======================================
# data list
# train
train_list = "label_train_list_fb.txt"
# dev
dev_list   = "label_dev_list_fb.txt"

mlf_file  = "../label/train.dev"

# basic configuration parameter
use_cuda = torch.cuda.is_available()
# network parameter 
dimension = 40 # 40 before
language_nums = 10  # 9!
learning_rate = 0.1
batch_size = 50
chunk_num = 10
#train_iteration = 10
train_iteration = 12
display_fre = 50
half = 4
# data augmentation
t1 = time.time()

phonemes_dict = dealMlf("../labels/train.mlf")
t2 = time.time()
print(t2 - t1)
def phonemeSeq(name_list):
    labels_sizes = []
    extension = '.fb'
    # the first sample
    name = name_list[0] + extension
    labels = phonemes_dict[name]
    labels_sizes.append(len(labels))

    # the other ones
    for name in name_list[1:]:
        name = name+extension
        arr = phonemes_dict[name]
        labels = np.concatenate((labels,arr))
        labels_sizes.append(len(arr))

    #labels_sizes = np.array(labels_sizes)
    return labels,labels_sizes

train_dataset = TorchDataSet(train_list, batch_size, chunk_num, dimension)

print(len(phonemes_dict))
for i in range(3):
    train_dataset.reset()

    for step, (batch_x, batch_y,name_list) in enumerate(train_dataset): 
        logging.info("step is %d"%(step))
        batch_target = batch_y[:,0].contiguous().view(-1, 1).long()
        batch_frames = batch_y[:,1].contiguous().view(-1, 1).long()
        #print(len(name_list),batch_x.shape,batch_target.shape)
        #print(np.array(name_list))
        labels , labels_sizes = phonemeSeq(name_list)
        #print(labels,labels_sizes)
        print(len(labels),sum(labels_sizes))

        if step == 9:
            break

