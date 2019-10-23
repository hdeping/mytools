#!/usr/bin/python

import openbabel as ob
import pybel 
import numpy as np

from getsmiles import getSMILES
from getsmiles import readSMILES
from getsmiles import readJson

import json


filename = "../inchikey_bonds.json"
# residue: bond information
residueBonds = readJson(filename)

filename = "../new_filter3.json"
# residue: "ID","molecule","type","energy"
residueEnergies  = readJson(filename)

filename = "../inchikey_filter_database.json"
# ID : "atoms","bonds","angles","dihedral"
idParameters  = readJson(filename)

# store the parameters
paraDicts = {}

"""
../energyNumber.json
../IDResidue.json
../inchikey_bonds.json
../inchikey_filter_database.json
../molEnergyNumber.json
../new_filter3.json
../residuesMatchDicts.json
"""

def stringToArr(string):
    res = []
    
    # deal with the string
    ch = string[2:-1]
    ch = ch.split(',')

    for item in ch:
        num = int(item)
        res.append(num)

    return res

def getAngles(indexC,indexO,parameters):
    angles = parameters["angles"]
    result = {}

    # find the exact angles
    for key in angles:
        arr = stringToArr(key)
        if arr[1] == indexC:
            if arr[0] == indexO or arr[2] == indexO:
                result[key] = angles[key]


    return result

def getDihedral(indexC,indexO,parameters):
    dihedral = parameters["dihedral"]
    result = {}

    # find the exact dihedral
    for key in dihedral:
        arr = stringToArr(key)
        if arr[1] == indexC and arr[2] == indexO:
            result[key] = dihedral[key]

    return result

def getBonds(indexC,indexO,parameters):
    bonds = parameters["bonds"]
    result = {}
    # first one should be C-O 
    key = "R(%d,%d)"%(indexC,indexO)
    if key not in bonds:
        para = json.dumps(parameters,indent = 4)
        #print(para)
        return False

    result[key] = bonds[key]

    for key in bonds:
        arr = stringToArr(key)
        # add to the result
        # R(x,indexC)
        if arr[1] == indexC:
            result[key] = bonds[key]
        # R(indexC,x) x != indexO
        if arr[0] == indexC and arr[1] != indexO:
            result[key] = bonds[key]
            

    return result

# get the bonds connected to "C"
count = 0
count_no_bond = 0
for residue in residueBonds:
    count += 1
    print("residue ",count)
    # index of the carbon and oxygen
    indexC = residueBonds[residue][0]
    indexO = residueBonds[residue][1]
    print(indexC,indexO)

    # get the id and molecule
    resDicts = residueEnergies[residue]
    id  = resDicts['ID']
    mol = resDicts['molecule']
    parameters = idParameters[id]
    

    #print(id)

    # get bonds
    molInfo = {}
    molInfo['ID'] = id
    molInfo['molecule'] = mol
    molInfo['type'] = resDicts['type']
    molInfo['energy'] = resDicts['energy']
    bonds = getBonds(indexC,indexO,parameters)

    # judge
    if bonds == False:
        print(id)
        print(mol)
        continue
        

    molInfo['bonds'] = bonds
    angles = getAngles(indexC,indexO,parameters)
    molInfo['angles'] = angles
    dihedral = getDihedral(indexC,indexO,parameters)
    molInfo['dihedral'] = dihedral

    paraDicts[residue] = molInfo

    #break

#paraDicts = json.dumps(paraDicts,indent = 4)
#print(paraDicts)
# write the data
fp = open("inchikey_parameters.json",'w')
json.dump(paraDicts,fp,indent = 4)


