#!/usr/bin/python


import openbabel as ob
import pybel 

string = "C1=CC=CC=C1"
mol = pybel.readstring("smi",string)
print(mol)
print(dir(mol))
print(mol.atoms)
for i in mol.atoms:
    print(i)

#filename = "input.smi"
#obConversion = ob.OBConversion()
#mol = ob.OBMol()
## read smiles
#obConversion.ReadFile(mol, filename)
# get bonds
MolBond = []
ChainBond = []

"""
for atom in ob.OBMolAtomIter(mol):
    print(atom.GetType())
#print(dir(atom))

for bond in ob.OBMolBondIter(mol):    
    a = bond.GetBeginAtomIdx()
    b = bond.GetEndAtomIdx()
    bo = bond.GetBondOrder()
    MolBond.append((a,b,bo))

    if bond.IsInRing() or bond.IsAromatic():
        continue

    ChainBond.append(bond.GetIdx())

#print(MolBond)
#MolBond.sort()
#print(MolBond)
#NumOfRingBond = len(MolBond) - len(ChainBond)



"""
