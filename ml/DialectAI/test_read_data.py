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

from read_data import get_samples, get_data, TorchDataSet
from net_component import LanNet

## ======================================
# 配置文件和参数
# 数据列表
train_list = "./label_train_list_fb.txt"
dev_list   = "./label_dev_list_fb.txt"

# 保存模型地址
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# 网络参数
dimension = 40
language_nums = 6
learning_rate = 0.1
batch_size = 64
chunk_num = 10
#train_iteration = 10
train_iteration = 30
display_fre = 50
half = 4


## ======================================
train_dataset = TorchDataSet(train_list, batch_size, chunk_num, dimension)
dev_dataset = TorchDataSet(dev_list, batch_size, chunk_num, dimension)
logging.info('finish reading all train data')
print(train_dataset._batch_size)
