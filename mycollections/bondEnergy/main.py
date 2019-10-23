#!/usr/bin/python


import openbabel as ob

filename = "input.smi"
obConversion = ob.OBConversion()
mol = ob.OBMol()
# read smiles
obConversion.ReadFile(mol, filename)
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
MolBond.sort()
print(MolBond)
NumOfRingBond = len(MolBond) - len(ChainBond)



