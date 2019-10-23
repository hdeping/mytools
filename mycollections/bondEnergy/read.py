#coding=utf-8
import numpy as np
import json

from getsmiles import readJson
from getsmiles import getSMILES

# get molecules
filename = "inchikey_bonds.json"
bonds = readJson(filename)

for mol in bonds:
    bond = bonds[mol]
    if bond[-1] != 'O':
        print(mol)
        

filename = "inchikey_bonds2.json"

