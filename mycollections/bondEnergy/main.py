#!/usr/bin/python


import openbabel as ob
import pybel 
import numpy as np

obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("smi", "smi")
vec = ob.OBBitVec()

def readSMILES(smi_string):
    # read smiles
    mol = ob.OBMol()
    obConversion.ReadString(mol, smi_string)
    new_mol = pybel.Molecule(mol)
    return mol


string1 = "[CH]OC(CC(=O)O)(CC(=O)O)O"
string2 = "[C](=O)C[C@@](CC(=O)O)(O)OC=O"
mol1 = readSMILES(string1)
mol2 = readSMILES(string2)

name = "FP2"
fingerprinter = ob.OBFingerprint.FindFingerprint(name)
vec1 = ob.vectorUnsignedInt()
vec2 = ob.vectorUnsignedInt()
fingerprinter.GetFingerprint(mol1, vec1)
fingerprinter.GetFingerprint(mol2, vec2)
tanimoto_value = fingerprinter.Tanimoto(vec1,vec2)
print(len(vec1))
print(tanimoto_value)
