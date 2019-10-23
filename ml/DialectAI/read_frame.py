import numpy as np


#name = 'label_dev_fb_frame.txt'
name = 'label_train_fb_frame.txt'
data = np.loadtxt('output',delimiter=' ',dtype=int)
print(data[:,-1])
res = np.zeros(10)
for i in data[:,-1]:
    res[i] += 1
print(res)
