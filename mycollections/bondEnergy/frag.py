#coding=utf-8
import openbabel as ob
import pybel as pb
import os
import numpy as np
import json

# import functions from the file getsmiles.py
from getsmiles import getSMILES
from getsmiles import getFingers
from getsmiles import FillAtom
from getsmiles import FragBondLink
from getsmiles import getBondInfo
from getsmiles import readSMILES


obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("smi", "smi")
# rematch the residues
filename = "rematch_residues.json"

def main(molecules,smi_string):

    # create a molecule container
    mol = readSMILES(smi_string)

    NumAtomsNoH = mol.NumAtoms()

    # get the bond information
    MolBond,ChainBond = getBondInfo(mol)

    # prepare to store fragment's SMILES
    ListOfFrag1 = []
    ListOfFrag2 = []
    atomId = []
    for BondIdx in range(len(MolBond)):
        obConversion.ReadString(mol, smi_string)
        #mol.AddHydrogens()
        a,b,bo = MolBond[BondIdx]
        # if A or B is oxygen
        AisO = mol.GetAtom(a).IsOxygen()
        BisO = mol.GetAtom(b).IsOxygen()
        AisC = mol.GetAtom(a).IsCarbon()
        BisC = mol.GetAtom(b).IsCarbon()
        #if (AisO or BisO) == False:
        #    print(a,b,False)
        #    continue
        #else:
        #    print(a,b,True)
        if (AisO or BisO) == False:
            continue
        if (AisC or BisC) == False:
            continue
        

        Frag1 = ob.OBMol()
        Frag1_idx = []
        Frag2 = ob.OBMol()
        Frag2_idx = []

        # find the begin atom and the end atom of the 
        # breaking bond. 
        a1 = mol.GetBond(BondIdx).GetBeginAtom()
        a2 = mol.GetBond(BondIdx).GetEndAtom()
        breakBO = mol.GetBond(BondIdx).GetBondOrder()
        # bonds information
        bond = mol.GetBond(BondIdx)


        # homolysis, create radical 
        a1.SetSpinMultiplicity(breakBO+1)
        a2.SetSpinMultiplicity(breakBO+1)
        mol.DeleteBond(mol.GetBond(BondIdx))

        if BondIdx in ChainBond:
            FillAtom(mol, a1, Frag1, Frag1_idx)
            FillAtom(mol, a2, Frag2, Frag2_idx)

            FragBondLink(Frag1, Frag1_idx, MolBond, NumAtomsNoH)
            FragBondLink(Frag2, Frag2_idx, MolBond, NumAtomsNoH)

            frag1_smi = pb.Molecule(Frag1).write("smi").replace('\t\n','')
            frag2_smi = pb.Molecule(Frag2).write("smi").replace('\t\n','')
        else:
            Frag1 = BreakRing(mol)
            frag2_smi = pb.Molecule(Frag1).write("smi").replace('\t\n','')
            frag1_smi = ''

        if frag2_smi == "[OH]" or frag2_smi == "[O]":
            ListOfFrag1.append(frag1_smi)
            ListOfFrag2.append(frag2_smi)
            # get bonds
            atomId.append(MolBond[BondIdx])
    #print("list 1")
    #print(ListOfFrag1)
    #print(ListOfFrag1,ListOfFrag2)
    #print("list 2")
    #print(ListOfFrag2)
    #print("bonds")
    #print(atomId)
    return len(atomId),ListOfFrag1


def readJson(filename):
    fp = open(filename,'r')
    molecules = json.load(fp)
    fp.close()

    return molecules

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
for i,smi_string in enumerate(filenames):
    #print("################# %d compounds #########"%(i))
    #fp.write("################# %d compounds #########"%(i)+'\n')
    #fp.write(smi_string + '\n')
    num,frag = main(molecules,smi_string)
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
        mismatch_frag = {}
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
            print(id)
            print(values)
            

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
count = count.astype(int)
count_mismatch = count_mismatch.astype(int)
count_mismatch_bond = count_mismatch_bond.astype(int)
print("freq",freq)
print("count",count)
print("total",total)
print("mismatch number is ",count_mismatch)
print("bond number  is ",count_mismatch_bond)
