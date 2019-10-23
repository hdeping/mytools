#coding=utf-8
import numpy as np
import json

from getsmiles import readJson
from getsmiles import getSMILES

# get molecules
filename = "inchikey_parameters.json"
bonds = readJson(filename)

for key in bonds:
    para = bonds[key]
    para = json.dumps(para,indent = 4)
    print(para)
    break
        

filename = "inchikey_bonds2.json"

