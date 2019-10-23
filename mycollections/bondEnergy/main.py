#!/usr/bin/python

import numpy as np

filename1 = "name2_2.txt"

# get dicts
dicts = {}
data1 = np.loadtxt(filename1,dtype=str)

for i in data1:
    dicts[i] = 0

# statistics
##########################
species = {}
species['[OH]'] = 0
species['[O]'] = 1
# get data from filter.csv
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
    # if this line is in the data file
    name = arr[1]
    i    = species[arr[3]]
    if name in data1:
        dicts[name] += i
            #print(arr[4])


##########################


# count
count = np.zeros(5)

for key in dicts:
    value = dicts[key]
    count[value] += 1

count = count.astype(int)
print(count)
print(sum(count))
