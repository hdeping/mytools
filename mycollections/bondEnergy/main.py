#!/usr/bin/python

import numpy as np
import json


def readJson(filename):
    fp = open(filename,'r')
    molecules = json.load(fp)
    fp.close()

    return molecules

def writeJson(filename,database):
    database = json.dumps(database,indent=4)
    fp = open(filename,'w')
    fp.write(database)
    fp.close()

filename = "../new_filter3.json"
molecules = readJson(filename)

id_dicts = {}
# initial 
#   id :[residues]
#   mol:[residues]
       
for residue in molecules:
    id  = molecules[residue]['ID']
    mol = molecules[residue]['molecule']
    id_dicts[id]  = []
    id_dicts[mol] = []

for residue in molecules:
    id  = molecules[residue]['ID']
    id_dicts[id].append(residue)


for residue in molecules:
    id  = molecules[residue]['ID']
    mol = molecules[residue]['molecule']
    id_dicts[mol] = id_dicts[id]


filename = "IDResidue.json"
writeJson(filename,id_dicts)
