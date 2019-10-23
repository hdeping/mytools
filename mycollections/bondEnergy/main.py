#!/usr/bin/python


import openbabel as ob

obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("smi", "smi")
mol = ob.OBMol()

# read smiles
smi_string = "C1=CC=CC=C1"

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
MolBond.sort()
print(MolBond)
NumOfRingBond = len(MolBond) - len(ChainBond)
