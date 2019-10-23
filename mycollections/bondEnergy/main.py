#!/usr/bin/python

import numpy as np
import json
import sys

def getSample(data,ii):
    result = {}
    # get atoms
    line = data[ii]
    keys = ["molecule", "type", "residue", "energy"]

    for i in range(3):
        key = keys[i]
        value = line[i+2]
        result[key] = value

    i = 3
    key = keys[i]
    value = line[i+2]
    result[key] = float(value)


    return result


##### check the files
#checkPara()

###############
"""
 name:(ID)
    molecule,
    type,
    residue,
    energy
"""
def getData(filename):
    fp = open(filename,'r')
    data = fp.read()
    fp.close()
    data = data.split('\n')
    data = data[:-1]
    res = []
    # get the first column
    for line in data:
        line = line.split(',')
        res.append(line)

    return res



filename = "../new_filter2.csv"
data = getData(filename)

size = len(data)
#print(data)
database = {}
for i in range(size):
    print("sample",i+1)
    result = getSample(data,i)
    name = data[i][1]
    database[name] = result

# dict to json
database = json.dumps(database,indent=4)

#### write the data
filename = "new_filter2.json"
fp = open(filename,'w')
fp.write(database)
fp.close()
