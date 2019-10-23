#coding=utf-8
import numpy as np
import json
import openbabel

from getsmiles import readJson
from getsmiles import getSMILES
from getsmiles import readSMILES

last_dir = "../"
# get molecules
filename = last_dir + "new_filter3.json"
molecules = readJson(filename)

filename = last_dir + "molEnergyNumber.json"
energyNum = readJson(filename)

filename = last_dir + "IDResidue.json"
idResidue = readJson(filename)

filename = last_dir + "inchikey_filter_database.json"
idResidue = readJson(filename)

filename = "../00_files/smiles.txt"
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

    print(filename)
    num = energyNum[filename]
    count += num

    mol = readSMILES(filename)
    a = mol.GetAtom(9)

    for atom in openbabel.OBAtomAtomIter(a):
        print(atom.GetType())
        bond = a.GetBond(atom)
        print("order",bond.GetBondOrder())

    break

print(count)



