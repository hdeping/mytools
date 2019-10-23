import torch
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.nn as nn
import numpy as np

a = [10,11,12,0,0,0]
b = [1,2,3,4,1,0]
c = [549,99,0,0,0,0]


src = torch.Tensor([a,b,c])
length = [3,5,2]
length = torch.LongTensor(length)
#new,sorted_indeces = torch.sort(length)
sorted_length,sorted_indeces = torch.sort(length,descending=True)
#print(sorted_indeces)
#print(sorted_length)

src = src[sorted_indeces]
src = src.unsqueeze(-1)
print(src.size())
pack = pack_padded_sequence(src,np.array(sorted_length),batch_first=True)
print(pack.data.size())
output,lengths = pad_packed_sequence(pack,batch_first=True)
print(output.size())

rnn = nn.GRU(input_size=1, hidden_size=3, batch_first=True, bidirectional=True)
out_hidden,hidd = rnn(pack)
out_hidden,lengths = pad_packed_sequence(out_hidden,batch_first=True)
print(out_hidden)
print(out_hidden.size())

out_hidden,hidd = rnn(src)
print(out_hidden)
print(out_hidden.size())
