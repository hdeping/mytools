#!/usr/bin/python

import numpy as np

filename1 = "exist.txt"
filenames = np.loadtxt(filename1,dtype=str)


def getPara(ii):
    filename = "../data/%s.para"%(filenames[ii])
    data = np.loadtxt(filename,delimiter=' ',dtype=str)
    return data
def checkPara():
    size = len(filenames)
    count = 0
    for i in range(size):
        data = getPara(i)
        # if the first letter is "R","A" or "D"
        letters = ["R","A","D"]
        for line in data[:,0]:
            first_letter = line[0]
            if first_letter not in letters :
                count = count + 1
                print(i)
                break

    print("there are %d wrong samples"%(count))

def getSerial(ii):
    filename = "../data/%s.serial"%(filenames[ii])
    data = np.loadtxt(filename,delimiter=' ',dtype=str)
    return data

ii = 0
data = getPara(ii)
print(data.shape)
data = getSerial(ii)
print(data.shape)

##### check the files
checkPara()

