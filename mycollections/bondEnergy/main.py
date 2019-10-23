#!/usr/bin/python


import openbabel as ob
import pybel 
import numpy as np

obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("smi", "smi")
#obFP = ob.OBFingerprint_Tanimoto()
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
tanimoto = ob.OBFingerprint.Tanimoto(mol1,mol2)
