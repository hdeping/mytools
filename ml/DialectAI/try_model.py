import torch

import torch.nn as nn

src = torch.randn(40,1,20,1000)
layer = nn.Conv2d(1,20,(3,3),stride=1,padding=1)

out = layer(src)
print(src.shape)
print(out.shape)
