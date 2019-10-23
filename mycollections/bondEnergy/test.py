#coding=utf-8
import numpy as np
import json

def getSMILES(filename):
    fp = open(filename,'r')
    data = fp.read()
    fp.close()
    data = data.split('\n')
    data = data[:-1]
    res = []
    # get the first column
    for line in data:
        line = line.split(' ')
        res.append(line[0])
    return res

def readJson(filename):
    fp = open(filename,'r')
    molecules = json.load(fp)
    fp.close()

    return molecules

# get molecules
filename = "../new_filter3.json"
molecules = readJson(filename)

filename = "../molEnergyNumber.json"
energyNum = readJson(filename)


filename = "../IDResidue.json"
idResidue = readJson(filename)

filename = "smiles.txt"
filenames = getSMILES(filename)

filename = "mismatch.result"
fp1 = open(filename,'w')
count_mismatch = np.zeros(9)

for i,smi_string in enumerate(filenames):
    residues = idResidue[smi_string]
    if len(residues) == 7:
        print(i,"here")
