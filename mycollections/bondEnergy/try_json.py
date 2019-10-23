#!/usr/bin/python

import json

# writing json files
def writing():
    dicts = {}
    dicts['a'] = 1
    dicts['b'] = 1.9
    dicts['c'] = 1.93434
    
    b = {}
    b['first one'] = dicts
    b = json.dumps(b,indent=4)
    fp = open("output.json",'w')
    fp.write(b)

def readJson(filename):
    fp = open(filename,'r')
    data = json.load(fp)
    print(data)
    #data = json.dumps(data,indent=4)
    return data

filename = "output.json"
data  = readJson(filename)
for key in data:
    print(key)
    value = data[key]
    for key1 in value:
        print(key1,value[key1])
