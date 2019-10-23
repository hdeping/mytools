# -*- coding:utf-8 -*-

import copy
import random
import numpy as np
from read_data import TorchDataSet

dev_list   = "../../labels/label_train_list_fb.txt"
batch_size = 50
chunk_num = 10
dimension = 40
p_vad = 0.2
#dev_dataset = TorchDataSet(dev_list, batch_size, chunk_num, dimension,p_vad)
dev_dataset = TorchDataSet(dev_list, batch_size, chunk_num, dimension)
for step, (batch_x, batch_y) in enumerate(dev_dataset): 
    print(step)
