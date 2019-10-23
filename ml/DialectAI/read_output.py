import numpy as np
import torch

data = np.loadtxt("output",delimiter=' ',dtype=int)
size = len(data) // 128
data = np.reshape(data,(size,128))
data1 = data[:,:64]
data2 = data[:,64:]
data1 = np.reshape(data1,(-1))
data2 = np.reshape(data2,(-1))

def old():
    print(data1.shape,data2.shape)
    count1 = 0
    count2 = 0
    data1  = torch.from_numpy(data1)
    data2  = torch.from_numpy(data2)
    new = data1.eq(data2)
    new = new.numpy()
    print(new)
    np.savetxt('new_out.txt',new,fmt='%d')
def new():
    res = []
    matrix = np.zeros((10,10))
    arr = np.zeros((10))
    size = len(data1)
    for i in range(size):
        ii = data1[i]
        jj = data2[i]
        arr[ii] += 1
        matrix[ii,jj] += 1
        if ii != jj:
            res.append([ii,jj])
    res = np.array(res)
    np.savetxt('new_out.txt',res,fmt='%d')
    matrix = matrix.astype(int)
    arr = arr.astype(int)
    print(matrix)
    print(arr)
    

new()
