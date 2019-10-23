#coding=utf-8

#2019-03-20 18:18:18
#by xiaohengdao

import numpy as np
import json

from getsmiles import readJson
from getsmiles import getSMILES

# get molecules
filename = "inchikey_bonds.json"
#"inchikey_parameters_CarbonOxygen.json"
#"inchikey_parameters_NoHydro.json"
#"inchikey_parameters_total.json"

names = ["inchikey_parameters_CarbonOxygen.json",
         "inchikey_parameters_NoHydro.json",
         "inchikey_parameters_total.json"]

bonds = readJson(names[1])

count = 0
for mol in bonds:
    count += 1

print(count)


