#!/usr/bin/python

import numpy as np

filename1 = "exist.txt"
filename2 = "total.txt"

# get dicts
dicts = {}
data1 = np.loadtxt(filename1,dtype=str)
data2 = np.loadtxt(filename2,dtype=str)

for i in data1:
    dicts[i] = 0

# statistics

for i in data2:
    dicts[i] += 1

# count
count = np.zeros(10)

for key in dicts:
    value = dicts[key]
    count[value] += 1

count = count.astype(int)
# repeats splitting
# get six files
filename = []
for i in range(6):
    name = 'name2_%d.txt'%(i+1)
    filename.append(name)
fp = []
for i in range(6):
    fp0 = open(filename[i],'w')
    fp.append(fp0)

for key in dicts:
    value = dicts[key]
    # value: 1-6
    fp[value-1].write(key+'\n')

res = 0
for i in range(10):
    print("number ",i,count[i])
    res += i*count[i]
print(sum(count))
print(res)

# close the file
for i in range(6):
    fp[i].close()
