#!/usr/bin/python


import openbabel as ob
import numpy as np

obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("smi", "smi")
mol = ob.OBMol()

def readSMILES(smi_string):
    # read smiles
    obConversion.ReadString(mol, smi_string)
    # get bonds
    MolBond = []
    ChainBond = []
    
    #for angle in ob.OBMolAngleIter(mol):    
    #    print(angle)
    for angle in ob.OBMolTorsionIter(mol):    
        print(angle)
    
    for bond in ob.OBMolBondIter(mol):    
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        bo = bond.GetBondOrder()
        MolBond.append((a,b,bo))
    
        if bond.IsInRing() or bond.IsAromatic():
            continue
    
        ChainBond.append(bond.GetIdx())
    
    #print(MolBond)
    #print("atom number",mol.NumAtoms())
    #print("bond number",mol.NumBonds())
    #print("residue number",mol.NumResidues())
    #MolBond.sort()
    #print(MolBond)
    #NumOfRingBond = len(MolBond) - len(ChainBond)
def getSMILES(filename):
    fp = open(filename,'r')
    data = fp.read()
    fp.close()
    data = data.split('\n')
    data = data[:-1]
    res = []
    # get the first column
    for line in data:
        line = line.split(' ')
        res.append(line[0])

    return res

    return

filename = "smiles.txt"
filenames = getSMILES(filename)
#print(filenames)
for smi_string in filenames:
    readSMILES(smi_string)
    break
    
