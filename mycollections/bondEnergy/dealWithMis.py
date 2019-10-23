#coding=utf-8
import numpy as np
import json

from getsmiles import readJson
from getsmiles import getSMILES
from getsmiles import getPairFingers

filename = "../../new_filter3.json"
molecules = readJson(filename)

def getMatch(matchInfo):
    length = len(matchInfo[0])
    # a,b,c,d,e to 0,1,2,3,4
    # ord('a') - ord('a')

    matchArr = []
    for line in matchInfo:
        arr = np.zeros(length,dtype=int)
        for match in line:
            match = match.split('-')
            index = int(match[0]) - 1
            num   = ord(match[1]) - ord('a')
            arr[index] = num
        # append to the arrays
        matchArr.append(arr)

    #print(matchArr)
    return matchArr

def getResidueList(matchDicts,ii):
    # ii = 2,3,4,5
    smi_file = "../02_files/mismatch_residues%d.smi"%(ii)
    match_file = "../02_files/mismatch_residues%d.txt"%(ii)

    # get data
    smiInfo = np.loadtxt(smi_file,dtype=str)
    matchInfo = np.loadtxt(match_file,dtype=str)
    # new shape
    smiInfo   = np.reshape(smiInfo,(-1,2,ii))
    matchInfo = np.reshape(matchInfo,(-1,ii))
    output    = getMatch(matchInfo)
    matchInfo = output
    sampleNum = len(smiInfo)
    # map the
    for i in range(sampleNum):
        smiSample = smiInfo[i]
        smiKey    = smiSample[0]
        smiValue  = smiSample[1]
        matchArr  = matchInfo[i]
        for j,key in enumerate(smiKey):
            index = matchArr[j]
            matchDicts[key] = smiValue[index]


    #for line in smiInfo:
    #    for i,item in enumerate(line):
    #        if item in molecules:
    #            print(i,item)

def getFingerMat(ii):
    smi_file = "../02_files/mismatch_residues_MapFingers%d.smi"%(ii)
    smiInfo = np.loadtxt(smi_file,dtype=str)
    smiInfo   = np.reshape(smiInfo,(-1,2,ii))
    
    for i,line in enumerate(smiInfo):
        for k1 in range(2):
            for k2 in range(ii):
                mol = line[k1,k2]
                if mol in molecules:
                    print(i,k1,k2,mol)
        """
        mat = getPairFingers(line[0],line[1])
        mat = np.array(mat)
        mat = mat.astype(int)
        num = np.sum(mat)
        if num != ii:
            print(i,line)
            
        """
            




for i in range(2,4):
    getFingerMat(i)
"""
filename = "../../IDResidue.json"
idResidue = readJson(filename)

filename = "../../new_filter3.json"
molecules = readJson(filename)

residuesMatchDicts = {}
for i in range(2,6):
    getResidueList(residuesMatchDicts,i)

residuesMatchDicts = json.dumps(residuesMatchDicts,indent=4)
print(residuesMatchDicts)
filename = "../02_files/residuesMatchDicts.json"
fp = open(filename,'w')
fp.write(residuesMatchDicts)
fp.close()

"""
