#coding=utf-8
import openbabel as ob
import os
import numpy as np
import json

# import functions from the file getsmiles.py
from getsmiles import getSMILES
from getsmiles import getFingers
from getsmiles import getFrag
from getsmiles import readJson
from getsmiles import printMismatchRes


obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("smi", "smi")

file_dir = "../02_files/"
last_dir = "../../"

# get molecules
filename = last_dir + "new_filter3.json"
molecules = readJson(filename)

filename = last_dir + "molEnergyNumber.json"
energyNum = readJson(filename)


filename = last_dir + "IDResidue.json"
idResidue = readJson(filename)

count = np.zeros(10)

#filename = "mismatch.txt"
filename = file_dir + "smiles.txt"
filenames = getSMILES(filename)

filename = last_dir + "residuesMatchDicts.json"
residuesMatchDicts = readJson(filename)
total = 0
freq = 0

count_mismatch = np.zeros(9)
count_mismatch_bond = np.zeros(9)

breakPoint = 0
breakPointNum = 999
count_mismatch_all = 0

filename = "smile_output.txt"
fp = open(filename,'w')
for i,smi_string in enumerate(filenames):
    print("################# %d compounds #########"%(i))
    #fp.write("################# %d compounds #########"%(i)+'\n')
    #fp.write(smi_string + '\n')
    num,frag = getFrag(molecules,smi_string)
    #print(i,energyNum[smi_string],num)
    real_num = energyNum[smi_string] 

    if num > real_num:
        print(num,real_num)

fp.close()

count = count.astype(int)
count_mismatch = count_mismatch.astype(int)
count_mismatch_bond = count_mismatch_bond.astype(int)
print("freq",freq)
print("count",count)
print("total",total)
print("mismatch number is ",count_mismatch)
print("bond number  is ",count_mismatch_bond)
print("mismatch all",count_mismatch_all)
