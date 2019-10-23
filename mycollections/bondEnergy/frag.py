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

file_dir = "../00_files/"
last_dir = "../"

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
filename = "symmetry_mismatch.txt"
fp1 = open(filename,'w')
counter_sym = 0
counter_sym_lost = 0

residueAtomsDicts = {}
for i,smi_string in enumerate(filenames):
    print("################# %d compounds #########"%(i))
    #fp.write("################# %d compounds #########"%(i)+'\n')
    #fp.write(smi_string + '\n')
    num, frag, fragDicts = getFrag(molecules,smi_string)
    #print(i,energyNum[smi_string],num)
    real_num = energyNum[smi_string] 

    residue = idResidue[smi_string][0]
    id  = molecules[residue]["ID"]


    # match_num
    match_num = 0
    if num > real_num:
        # some items are repeated
        # mismatched ones can be ignored
        #print("compounds %d %s, real num %d, num %d"%(i,smi_string,real_num,num))
        count_exist = 0
        for line in frag:
            if line in molecules:
                fp.write(line +'\n')
                mol = molecules[line]["molecule"]
                count_exist += 1
                # write to the dicts
                residueAtomsDicts[line] = fragDicts[line]

        if count_exist != real_num:
            print(count_exist,real_num)
            print("oh!!!! number mismatch here!")
            fp1.write("%s exist number %d  real number %d\r"%(id,count_exist,real_num))
            #break

        match_num = num
        counter_sym += 1
        counter_sym_lost += (real_num - count_exist)
    else:
        # some smiles are displayed into diffrent formulas
        # which should be equivalent

        ####  get residues
        residues = idResidue[smi_string]

        ### match the exact samples
        mismatch_frag = []
        for line in frag:
            #print("line",line)
            if line in molecules:
                a = (line in residues)
                #id = molecules[line]["ID"]
                exactLine = line
                if not a:
                    #print("nooooooo",line,molecules[line]['ID'],residues)
                    break
                #id = molecules[line]["ID"]
                fp.write(line +'\n')
                bond = fragDicts[line]
                residueAtomsDicts[line] = bond
                print(bond)
                #print(line,residues)
                residues.remove(line)
                match_num += 1
            else:
                mismatch_frag.append(line)

        #print(num,real_num,match_num,len(residues))



        #if there is only one residue in the residues
        if len(residues) == 1:
            print(len(mismatch_frag))
            fp.write(residues[0] + '\n')
            key1 = residues[0]
            key2 = mismatch_frag[0]
            bond = fragDicts[key2]
            residueAtomsDicts[key1] = bond
            print(bond)
            match_num = num 

    if num - match_num > 0:
        judge = 0
        for line in mismatch_frag:
            if line not in residuesMatchDicts:
                #print(i,id,line,len(mismatch_frag),"wrong here")
                judge = 1
                print("------------")
                #break
            else:
                key = residuesMatchDicts[line]
                bond = fragDicts[line]
                residueAtomsDicts[key] = bond
                print(bond)

        if judge == 1:
            for line in mismatch_frag:
                #print(line)
                if line in residuesMatchDicts:
                    print("newererererererer")
                    print(residuesMatchDicts[line])

    count_mismatch[num - match_num] += 1
    if num - match_num == 1:
        id = molecules[exactLine]["ID"]
        residues = idResidue[id]
        print("frag num: %d, residues num: %d, real num: %d"%(len(frag),len(residues),real_num))
        print("num",num,"match num",match_num,frag,residues)
        break
        
    #if num - match_num == 4:
    #    print(smi_string)
    #    print(frag)
    #    #break
    
    total += num
    count[num] += 1
    freq += 1
    if freq == 20000:
        break

fp.close()
fp1.close()

count = count.astype(int)
count_mismatch = count_mismatch.astype(int)
count_mismatch_bond = count_mismatch_bond.astype(int)
print("freq",freq)
print("count",count)
print("total",total)
print("mismatch number is ",count_mismatch)
print("bond number  is ",count_mismatch_bond)
print("mismatch all",count_mismatch_all)
print("count symmetry",counter_sym)
print("count symmetry number",counter_sym_lost)

# print the bonds
filename = "inchikey_bonds.json"
fp = open(filename,'w')
json.dump(residueAtomsDicts,fp,indent=4)
fp.close()
