#coding=utf-8
import numpy as np
import json

from getsmiles import readJson
from getsmiles import getSMILES

# get molecules
filename = "../../new_filter3.json"
molecules = readJson(filename)

filename = "../../molEnergyNumber.json"
energyNum = readJson(filename)

filename = "../../IDResidue.json"
idResidue = readJson(filename)

filename = "../../inchikey_filter_database.json"
idResidue = readJson(filename)

filename = "../02_files/smiles.txt"
filenames = getSMILES(filename)

count_mismatch = np.zeros(9)


def getOutputSmiles():
    count = np.zeros(5,dtype=int)
    output = {}
    filename = "smile_output.txt"
    filenames = getSMILES(filename)
    for filename in filenames:
        output[filename] = []
    for i,filename in enumerate(filenames):
        output[filename].append(i)
    repeatOut = {}
    for key in output:
        num = len(output[key]) 
        count[num] += 1
        if num > 1:
            repeatOut[key] = output[key]


    print("output",count)
    print(len(output))

    filename = "smile_output.json"
    fp = open(filename,'w')
    filename = "smile_repeatOutput.json"
    fp1 = open(filename,'w')

    json.dump(output,fp,indent=4)
    json.dump(repeatOut,fp1,indent=4)

    fp.close()
    fp1.close()

    return output



"""
for i,smi_string in enumerate(filenames):
    residues = idResidue[smi_string]
    print(i,residues)
    if len(residues) == 7:
        print(i,"here")

"""
count = 0
# frag output
ouput = getOutputSmiles()
for filename in filenames:

    num = energyNum[filename]
    count += num

print(count)



