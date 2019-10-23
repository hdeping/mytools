#!/usr/bin/python

import numpy as np

name = "output.txt"
data = np.loadtxt(name,delimiter=' ',dtype=float)
print(data)

arr0 = []
arr1 = []
for line in data:
    if line[0] == line[1]:
        arr0.append(line[2])
    else:
        arr1.append(line[2])

arr0 = np.array(arr0)
arr1 = np.array(arr1)

np.savetxt("00_right.txt",arr0)
np.savetxt("00_wrong.txt",arr1)

def printArrStati(arr):
    total  = sum(arr)
    length = len(arr)
    print(total,total/length)

printArrStati(arr0)
printArrStati(arr1)

