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

    atoms = []
    # find the exact angles
    for key in angles:
        arr = stringToArr(key)
        if arr[1] == indexC:
            if arr[0] == indexO or arr[2] == indexO:
                result[key] = angles[key]
                atoms.append(arr)


    return result,atoms

def getDihedral(atoms,parameters):
    dihedral = parameters["dihedral"]
    result = {}

    # find the exact dihedral
    for key in dihedral:
        arr = stringToArr(key)
        #if arr[1] == indexC and arr[2] == indexO:
        for angle in atoms:
            print(angle)
            # if the three atoms of the angles 
            # are all in the dihedrals
            p = []
            for i in range(3):
                p1 = (angle[i] in arr)
                p.append(p1)
            #if angle[0] in arr and angle[1] in arr and angle[2] in arr:
            if p[0] and p[1] and p[2]:
                result[key] = dihedral[key]

    return result

def getBonds(indexC,indexO,parameters):
    #print("iteration")
    bonds = parameters["bonds"]
    print(bonds)
    result = {}
    # store the bonds
    atoms = []
    # first one should be C-O 
    key = "R(%d,%d)"%(indexC,indexO)
    atoms.append([indexC,indexO])
    if key not in bonds:
        para = json.dumps(parameters,indent = 4)
        #print(para)
        return False,atoms

    result[key] = bonds[key]

    #print("iteration")
    for key in bonds:
        arr = stringToArr(key)
        # add to the result
        # R(x,indexC)
        if arr[1] == indexC:
            result[key] = bonds[key]
            atoms.append([arr[0],indexC])
        # R(indexC,x) x != indexO
        if arr[0] == indexC and arr[1] != indexO:
            result[key] = bonds[key]
            atoms.append([indexC,arr[1]])
            

    print(atoms)
    return result,atoms

# get the bonds connected to "C"
count = 0
count_no_bond = 0
count_sample = 0
dihedral_counts = np.zeros(20,dtype=int)
for residue in residueBonds:
    count += 1
    print("residue ",count)
    # index of the carbon and oxygen
    indexC = residueBonds[residue][0]
    indexO = residueBonds[residue][1]
    #print(indexC,indexO)

    # get the id and molecule
    resDicts = residueEnergies[residue]
    id  = resDicts['ID']
    mol = resDicts['molecule']
    energy = resDicts['energy']
    parameters = idParameters[id]
    

    #print(id)

    # get bonds
    molInfo = {}
    molInfo['ID'] = id
    molInfo['molecule'] = mol
    molInfo['type'] = resDicts['type']
    molInfo['energy'] = resDicts['energy']

    bonds,atoms = getBonds(indexC,indexO,parameters)

    # judge
    if bonds == False:
        print(id)
        print(mol)
        continue
        

    molInfo['bonds'] = bonds
    angles,atoms = getAngles(indexC,indexO,parameters)
    molInfo['angles'] = angles
    dihedral = getDihedral(atoms,parameters)
    molInfo['dihedral'] = dihedral

    filter1 = (molInfo['type'] == '[O]')
    bondNum = 0
    for bond in bonds:
        bondNum += 1
    angleNum = 0
    for angle in angles:
        angleNum += 1
    dihedralNum = 0
    for dihed in dihedral:
        dihedralNum += 1
    if dihedralNum == 10:
        print(dihedral)

    dihedral_counts[dihedralNum] += 1
    filter2 = (bondNum  == 3)
    filter3 = (angleNum == 2)
    
        
    if filter1 and filter2 and filter3:
        paraDicts[residue] = molInfo
        count_sample += 1

    #break

#paraDicts = json.dumps(paraDicts,indent = 4)
#print(paraDicts)
# write the data
fp = open("inchikey_parameters.json",'w')
json.dump(paraDicts,fp,indent = 4)


print("number of the samples ", count_sample)

print("dihedral statistics")
print(dihedral_counts)
print(sum(dihedral_counts))
