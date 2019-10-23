#!/usr/bin/python

import numpy as np

filename = "../new_filter2.csv"

fp = open(filename,'r')
data = fp.read()

data = data.split('\n')
data = data[:-1]
#print(data)
#print(len(data))

count = 0
for line in data:
    arr = line.split(',')
    if arr[3] != '[O-]':
        #print(arr[4])
        print(line)

#print(count)


