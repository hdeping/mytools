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
    envDicts[key[:2]] = 0

envAtoms = {}
for key in bonds:
    para = bonds[key]
    #para = json.dumps(para,indent = 4)
    res =  getHash(para)
    envDicts[str(res // 10)] += 1
    #print(para['bonds']['atoms'])
    #print(res)
    atomSeq = para['bonds']['atoms']
    envAtoms[atomSeq] = 0
    #break
        
for key in bonds:
    para = bonds[key]
    atomSeq = para['bonds']['atoms']
    """
    if atomSeq == "COHCCCC":
        print(json.dumps(para,indent = 4))
    """
    envAtoms[atomSeq] += 1

envDicts = json.dumps(envDicts,indent = 4)
envAtoms = json.dumps(envAtoms,indent = 4)
print(envDicts)
print(envAtoms)

