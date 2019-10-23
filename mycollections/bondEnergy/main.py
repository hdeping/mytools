#!/usr/bin/python

import numpy as np
import json
import sys

filename1 = sys.argv[1] 
filenames = np.loadtxt(filename1,dtype=str)
suffix = '_frag00.out'


def getPara(ii):
    filename = "../data/%s%s.para"%(filenames[ii],suffix)
    data = np.loadtxt(filename,delimiter=' ',dtype=str)

    numberR = 0
    numberA = 0
    # order: R,A,D
    for line in data[:,0]:
        first_letter = line[0]
        if first_letter == "R":
            numberR += 1
        elif first_letter == "A":
            numberA += 1

    # bonds,angles,dihedral
    bonds    = data[:numberR]
    angles   = data[numberR:numberR+numberA]
    dihedral = data[numberR+numberA:]

    return bonds,angles,dihedral

def checkPara():
    size = len(filenames)
    count = 0
    for i in range(size):
        data = getPara(i)
        # if the first letter is "R","A" or "D"
        letters = ["R","A","D"]
        #print(data[:,0])
        for line in data[:,0]:
            first_letter = line[0]
            #print(first_letter)
            if first_letter not in letters :
                count = count + 1
                print(i,first_letter)
                break

        #break

    print("there are %d wrong samples"%(count))

def getSerial(ii):
    filename = "../data/%s%s.serial"%(filenames[ii],suffix)
    data = np.loadtxt(filename,delimiter=' ',dtype=str)
    return data

def arrToDicts(arr):
    res = {}
    # arr to dicts

    for line in arr:
        key = line[1]
        value = float(line[2])
        res[key]  = value

    return res
def getSample(ii):
    result = {}
    # get atoms
    serial = getSerial(ii)
    result["atoms"] = list(serial[:,0])

    # get bonds, angles, dihedral
    bonds, angles, dihedral = getPara(ii)
    #print(bonds)
    #print(angles)
    #print(dihedral)

    result["bonds"] = arrToDicts(bonds)
    result["angles"] = arrToDicts(angles)
    result["dihedral"] = arrToDicts(dihedral)

    return result


##### check the files
#checkPara()

###############
"""
 name:
   atoms: str,list
   bonds:
        id: str,list
        pair: array (n)
        value: list,float
   angles:
        id: str,list
        pair: int,list ( 3 or 5)
        value: float,list
   dihedral:
        id: str,list
        pair: int,list (n,4)
        value: float,list
"""
size = len(filenames)
#size = 100
database = {}
for i in range(size):
    print("sample",i+1)
    result = getSample(i)
    name = filenames[i]
    database[name] = result

# dict to json
database = json.dumps(database,indent=4)

#### write the data
filename = "inchikey_filter_database%s.json"%(sys.argv[1])
filename = filename.replace("name2",'')
filename = filename.replace(".txt",'')
fp = open(filename,'w')
fp.write(database)
fp.close()
