#coding=utf-8
import numpy as np
import json

from getsmiles import readJson
from getsmiles import getSMILES

# get molecules
filename = "inchikey_parameters.json"
bonds = readJson(filename)

# get bond type
filename  =  "type.json"
bondTypes = readJson(filename)
i = 0
for key in bondTypes:
    bondTypes[key] = i
    i = i + 1

# C=O
carbonyl = []
# C-OH
hydroxyl = []
def getHash(para):
    number_bonds     = para['bonds']['number']
    number_angles    = para['angles']['number']
    number_dihedrals = para['dihedral']['number']
    list = [number_bonds,number_angles,number_dihedrals]

    res = 0
    for ii in list:
        res = 10*res + ii

    return res


envDicts = {}
keys = ["200","210","220","222","233",
        "310","320","321","322","333",
        "420","430","433","533"]
for key in keys:
    #envDicts[key[:2]] = 0
    envDicts[key] = 0

def paraToArr(para):
    # 4 bonds and 3 angles and 1 energy
    # 3 bonds: the 4th one would be zero
    # the same to the angle
    res = []
    bonds = para["bonds"]
    angles = para["angles"]

    # bond type
    atomSeq = para['bonds']['atoms']
    res.append(bondTypes[atomSeq])
    # 3 bonds or 4 bonds
    for key in bonds:
        if key[0] == 'R':
            bondLength = bonds[key]
            res.append(bondLength)
    if bonds['number'] == 3:
        res.append(0)


    # 2 angles or 3 bonds
    for key in angles:
        if key[0] == 'A':
            angle = angles[key]
            res.append(angle)
    if angles['number'] == 2:
        res.append(0)

    # 1 energy
    energy = para["energy"]
    res.append(energy)

    return res

COCC_data = {}
# parameter array
COCC_arr = []
count = 0

for key in bonds:
    para = bonds[key]
    #atomSeq = para['bonds']['atoms']
    #if atomSeq == "COCC":

    res = getHash(para)
    envDicts[str(res)] += 1
    if res // 10 == 32 or res // 10 == 43:
        COCC_data[key] = para
        arr = paraToArr(para)
        COCC_arr.append(arr)
        count += 1
            
        

filename = "COCC_data.csv"
COCC_arr = np.array(COCC_arr)
np.savetxt(filename,COCC_arr,fmt="%f")

print(count)

envDicts = json.dumps(envDicts,indent = 4)
print(envDicts)
