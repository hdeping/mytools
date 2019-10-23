# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

inf=-float('inf')
a = [[1,1,1,inf],[-1,2,1,inf],[3,3,3,inf]]
a = torch.Tensor(a)
print(a)
b = F.softmax(a,dim=1)
print(b)
