import numpy as np
#name = 'label_dev_fb_frame.txt'
name = 'output'
data = np.loadtxt(name,delimiter=' ',dtype=str)
np.random.shuffle(data)
print(data[:,-1])
res = np.zeros(10)
for i in data[:,-1]:
    res[int(i)] += 1
print(res)
train_list = "../labels/label_train_fb_frame.txt"
data = np.loadtxt(train_list,delimiter=' ',dtype=str)
for i in range(10):
    ii = i*6000
    #print(data[ii,-1])
    ii = data[ii,-1]
    print(res[int(ii)]/60000)
