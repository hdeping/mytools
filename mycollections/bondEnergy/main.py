#!/usr/bin/python

import numpy as np
import json

filename1 = "exist.txt"
filenames = np.loadtxt(filename1,dtype=str)


def getPara(ii):
    filename = "../data/%s.para"%(filenames[ii])
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
    filename = "../data/%s.serial"%(filenames[ii])
    data = np.loadtxt(filename,delimiter=' ',dtype=str)
    return data

# arr (n,3)
def arrToDicts(arr):
    res = {}
    res["ID"]    = list(arr[:,0])
    res["pair"]  = list(arr[:,1])
    res["value"] = list(arr[:,2])

    return res
def getSample(ii):
    result = {}
    # get atoms
    serial = getSerial(ii)
    result["atoms"] = list(serial[:,0])

    bonds, angles, dihedral = getPara(ii)
    #print(bonds)
    #print(angles)
    #print(dihedral)

    # get bonds
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
result = getSample(0)
result = json.dumps(result,indent=4)
print(result)
