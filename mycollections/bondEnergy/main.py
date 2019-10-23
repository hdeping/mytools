#!/usr/bin/python


import openbabel as ob
import pybel 
import numpy as np

from getsmiles import getSMILES

obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("smi", "smi")

def readSMILES(smi_string):
    # read smiles
    mol = ob.OBMol()
    obConversion.ReadString(mol, smi_string)
    new_mol = pybel.Molecule(mol)
    return mol
def getFingers(mols):
    vec1 = ob.vectorUnsignedInt()
    vec2 = ob.vectorUnsignedInt()
    fingerprinter.GetFingerprint(mols[0], vec1)
    
    for mol in mols:
        fingerprinter.GetFingerprint(mol, vec2)
        tanimoto_value = fingerprinter.Tanimoto(vec1,vec2)
        print(tanimoto_value)

def getAllMols(filename):
    strings = getSMILES(filename)
    mols = []
    for name in strings:
        mol = readSMILES(name)
        mols.append(mol)

    return mols


filename = "03_sixRepeats.smi"
# get all smiles of the molecules
mols = getAllMols(filename)
getFingers(mols)

