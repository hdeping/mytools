import torch
a = [[1,3,2,4],
     [5,7,3,9],
     [3,3,4,8]]
a = torch.Tensor(a)
print(a)
b = a.min(dim=1)
print(b)
