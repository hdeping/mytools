#coding=utf-8
import numpy as np
import json

from getsmiles import readJson
from getsmiles import getSMILES

# get molecules
filename = "inchikey_parameters.json"
bonds = readJson(filename)

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

COCC_data = {}
for key in bonds:
    para = bonds[key]
    atomSeq = para['bonds']['atoms']
    if atomSeq == "COCC":
        res = getHash(para)
        envDicts[str(res)] += 1
        if res // 10 == 32 :
            COCC_data[key] = para
            print(key)
            
        
        

envDicts = json.dumps(envDicts,indent = 4)
print(envDicts)

#envAtoms = json.dumps(envAtoms,indent = 4)
#print(envAtoms)

filename = "COCC_data.json"
fp = open(filename,'w')
json.dump(COCC_data,fp,indent = 4)

