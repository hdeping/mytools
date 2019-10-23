#!/usr/bin/python

import numpy as np

#filename = "../new_filter2.csv"
filename = "filter.csv"
data = np.loadtxt(filename,delimiter=' ',dtype=str)
print(data)
