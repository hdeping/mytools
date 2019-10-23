#!/usr/bin/python

import numpy as np

filename1 = "exist.txt"
filename2 = "total.txt"

def getSMILES(filename):
    fp = open(filename,'r')
    data = fp.read()
    fp.close()
    data = data.split('\n')
    data = data[:-1]
    return data
# get dicts
dicts = {}
data1 = getSMILES(filename1)
data2 = getSMILES(filename2)

#print(data1[6405:6415])
print(len(data1))
print(len(data2))

for index,i in enumerate(data1):
    if len(i) == 1:
        print(index,"length",len(i))
    dicts[i] = 0

# statistics

for i in data2:
    dicts[i] += 1

# count
count = np.zeros(10)

for key in dicts:
    value = dicts[key]
    #print(key,value)
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
