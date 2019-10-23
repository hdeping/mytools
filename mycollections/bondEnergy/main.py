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
    
    
    for bond in ob.OBMolBondIter(mol):    
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        bo = bond.GetBondOrder()
        MolBond.append((a,b,bo))
    
        if bond.IsInRing() or bond.IsAromatic():
            continue
    
        ChainBond.append(bond.GetIdx())
    
    print(MolBond)
    #MolBond.sort()
    #print(MolBond)
    #NumOfRingBond = len(MolBond) - len(ChainBond)

filename = "smiles.txt"
filenames = np.loadtxt(filename,delimiter=' ',dtype=str)
for smi_string in filenames[:,0]:
    readSMILES(smi_string)
    
