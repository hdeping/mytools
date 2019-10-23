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
    #print(key,value)
    count[value] += 1

count = count.astype(int)
# six repeats
for key in dicts:
    value = dicts[key]
    if value == 6:
        print(key)

res = 0
for i in range(10):
    print("number ",i,count[i])
    res += i*count[i]
print(sum(count))
print(res)

import json
dicts = json.dumps(dicts,indent=4)
filename = "energyNumber.json"
fp = open(filename,'w')
fp.write(dicts)
fp.close()
