#!/usr/bin/python

import numpy as np
import json
import sys

def getSample(data,ii):
    result = {}
    # get atoms
    line = data[ii]
    keys = ["ID","molecule", "type","energy"]

    result[keys[0]] = line[1]
    result[keys[1]] = line[2]
    result[keys[2]] = line[3]
    result[keys[3]] = float(line[5])


    return result


##### check the files
#checkPara()

###############
"""
 residue:
    ID,
    molecule,
    type,
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
    name = data[i][4]
    database[name] = result

# dict to json
database = json.dumps(database,indent=4)

#### write the data
filename = "new_filter2.json"
fp = open(filename,'w')
fp.write(database)
fp.close()
