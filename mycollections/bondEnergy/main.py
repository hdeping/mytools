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
# ID : "atoms","bonds","angles","dihedrals"
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


def getAngles(indexC,indexO,mol):
    angles = []
    for angle in ob.OBMolAngleIter(mol):
        print(angle)
        if angle[1] == indexC - 1:
            if angle[0] == indexO - 1 or angle[2] == indexO - 1:
                string = "A(%d,%d,%d)"%(angle[0]+1,angle[1]+1,angle[2]+1)
                angles.append(string)

    return angles

def getBonds(indexC,indexO,mol):

    # get the neighbor of the carbon
    carbon = mol.GetAtom(indexC)
    for carbonNeighbor in ob.OBAtomAtomIter(carbon):
        print(carbonNeighbor.GetIdx())

# get the bonds connected to "C"
for residue in residueBonds:
    # index of the carbon and oxygen
    indexC = residueBonds[residue][0]
    indexO = residueBonds[residue][1]
    print(indexC,indexO)
    # get the id and molecule
    id = residueEnergies[residue]['ID']
    mol = residueEnergies[residue]['molecule']
    print(mol)
    # get the parameters
    para = idParameters[id]
    para = json.dumps(para,indent = 4)
    print(para)
    mol = readSMILES(mol)
    # add hydrogens
    mol.AddHydrogens()
    

    print(id)

    # get bonds
    bonds = getBonds(indexC,indexO,mol)

    angles = getAngles(indexC,indexO,mol)
    print(angles)

    break

# write the data
# fp = open("inchikey_parameters.json",'w')
# json.dump(paraDicts,fp,indent = 4)


