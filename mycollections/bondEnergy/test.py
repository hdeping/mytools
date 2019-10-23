#!/usr/bin/python


filename  = "output"

import numpy as np

data = np.loadtxt(filename,dtype=str)

print(len(data))
for i in range(len(data)):
    if len(data[i]) == 27:
        print(data[i])
