#!/usr/bin/python
import openbabel as ob

# formats of the fingerprints
finger_formats = ["FP2", "FP3", "FP4", "MACCS","ECFP0" , "ECFP10", "ECFP2", "ECFP4", "ECFP6", "ECFP8"]
name = finger_formats[2]
#name = "fpt"
fingerprinter = ob.OBFingerprint.FindFingerprint(name)


def getFingers(mols):
    vec1 = ob.vectorUnsignedInt()
    vec2 = ob.vectorUnsignedInt()
    fingerprinter.GetFingerprint(mols[0], vec1)
    
    values = []
    for mol in mols:
        fingerprinter.GetFingerprint(mol, vec2)
        tanimoto_value = fingerprinter.Tanimoto(vec1,vec2)
        values.append(tanimoto_value)

    # print the tanimoto values
    print(values)

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

