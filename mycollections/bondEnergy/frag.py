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


obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("smi", "smi")
# rematch the residues
filename = "rematch_residues.json"


# get molecules
filename = "../new_filter3.json"
molecules = readJson(filename)

filename = "../molEnergyNumber.json"
energyNum = readJson(filename)


filename = "../IDResidue.json"
idResidue = readJson(filename)

count = np.zeros(10)

#filename = "mismatch.txt"
filename = "smiles.txt"
filenames = getSMILES(filename)
total = 0
freq = 0

filename = "output.result"
fp = open(filename,'w')
filename = "mismatch.result"
fp1 = open(filename,'w')
count_mismatch = np.zeros(9)
count_mismatch_bond = np.zeros(9)

# get mismatched  and finger-repeated residues
filename = "mismatch_residues.smi"
fp2 = open(filename,'w')
for i,smi_string in enumerate(filenames):
    #print("################# %d compounds #########"%(i))
    #fp.write("################# %d compounds #########"%(i)+'\n')
    #fp.write(smi_string + '\n')
    num,frag = getFrag(molecules,smi_string)
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
        for line in frag:
            if line in molecules:
                fp.write(line +'\n')
                #mol = molecules[line]["molecule"]
        num = real_num
        match_num = num
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
                #print(line,residues)
                residues.remove(line)
                match_num += 1
            else:
                mismatch_frag.append(line)

        #print(num,real_num,match_num,len(residues))



        #if there is only one residue in the residues
        if len(residues) == 1:
            fp.write(residues[0] + '\n')
            match_num = num 

    # count the mismatch number
    #print(num,match_num)
    if num - match_num > 0:
        mol = molecules[exactLine]["molecule"]
        count_mismatch_bond[num] += 1
        fp1.write(mol + '\n')
        # print the fingerprints of the mismatched residues
        result,values = getFingers(residues)
        if result:
            print("% ",id)
            fp2.write("%s %s\n"%('%',id))
            print(mismatch_frag)
            for line in mismatch_frag:
                fp2.write("%s\n"%(line))
            for line in residues:
                fp2.write("%s\n"%(line))
            print(residues)
            if len(mismatch_frag) != len(values):
                print("match is wrong here!!!!!")
                break
            

    count_mismatch[num - match_num] += 1
    if num - match_num == 1:
        id = molecules[exactLine]["ID"]
        residues = idResidue[id]
        print(num,match_num,frag,residues)
        #break
        
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
fp2.close()
count = count.astype(int)
count_mismatch = count_mismatch.astype(int)
count_mismatch_bond = count_mismatch_bond.astype(int)
print("freq",freq)
print("count",count)
print("total",total)
print("mismatch number is ",count_mismatch)
print("bond number  is ",count_mismatch_bond)
