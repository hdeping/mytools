#!/usr/bin/python
import openbabel as ob


obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("smi", "smi")

# formats of the fingerprints
finger_formats = ["FP2", "FP3", "FP4", "MACCS","ECFP0" , "ECFP10", "ECFP2", "ECFP4", "ECFP6", "ECFP8"]
name = finger_formats[2]
#name = "fpt"
fingerprinter = ob.OBFingerprint.FindFingerprint(name)

def readSMILES(smi_string):
    # read smiles
    mol = ob.OBMol()
    obConversion.ReadString(mol, smi_string)
    return mol
def getAllMols(strings):
    mols = []
    for name in strings:
        mol = readSMILES(name)
        mols.append(mol)

    return mols


# judge if there are repeat elements
# in a list
def IsRepeat(input_list):
    num = len(input_list)

    for i in range(num-1):
        ii = input_list[i]
        jj = input_list[i+1]
        if ii == jj:
            return True
    
    return False



def getFingers(strings):
    # get molecules
    mols = getAllMols(strings)
    vec1 = ob.vectorUnsignedInt()
    vec2 = ob.vectorUnsignedInt()
    fingerprinter.GetFingerprint(mols[0], vec1)
    
    values = []
    for mol in mols:
        fingerprinter.GetFingerprint(mol, vec2)
        tanimoto_value = fingerprinter.Tanimoto(vec1,vec2)
        values.append(tanimoto_value)

    # print the tanimoto values
    if IsRepeat(values):
        print("repeating",values)

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

# Break the bond in ring and generate the radical fragment
def BreakRing(mol):
    ring_frag = ob.OBMol()
    for atom in ob.OBMolAtomIter(mol):
        ring_frag.AddAtom(atom)
    for bond in ob.OBMolBondIter(mol):
        ring_frag.AddBond(bond)
    return ring_frag

# Determinate the second-end atom
def IsNearTerminal(atom):
    n = 0
    if atom.GetSpinMultiplicity() == 0:
        for _atom in ob.OBAtomAtomIter(atom):
            if _atom.GetType() in ["H", "F", "Cl", "Br", "I"]:
                continue
            else:
                n = n + 1
        return n == 1
    else: 
        return False

# FillAtom() is a function to find all the atom in fragment
def FillAtom(mol, atom, frag, fatom_idx):
    # mol, frag -- Class OBMol
    # atom -- Class OBAtom
    # fatom_idx -- Record the atom has existed
    frag.AddAtom(atom)
    fatom_idx.append(atom.GetIdx())
    if atom.GetValence == 0:
        return frag
    elif IsNearTerminal(atom):
        for _atom in ob.OBAtomAtomIter(atom):
            index = _atom.GetIdx()
            if index in fatom_idx: continue
            frag.AddAtom(_atom)
            fatom_idx.append(index)
        return frag
    else:
        for _atom in ob.OBAtomAtomIter(atom):
            index = _atom.GetIdx()
            if index in fatom_idx: continue
            elif _atom.GetValence == 1:
                frag.AddAtom(_atom)
                fatom_idx.append(index)
            else:
                FillAtom(mol, _atom, frag, fatom_idx)

# BondLink() is a function to bonding the atoms in the fragment 
def FragBondLink(frag, fatom_idx,mol_bond,NumAtomsNoH):
    IdxDict = {}
    for i in range(len(fatom_idx)):
        IdxDict[fatom_idx[i]] = i+1
    
    n = 0
    for j in range(NumAtomsNoH,0,-1):
        if j in fatom_idx:
            n = j
            break
    for BAtom, EAtom, BO in mol_bond:
        if BAtom > n:
            break
        try:
            frag.AddBond(IdxDict.get(BAtom), IdxDict.get(EAtom), BO)
        except:
            continue

def getBondInfo(mol):
    # get the bond information
    MolBond = []
    ChainBond = []
    for bond in ob.OBMolBondIter(mol):    
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        bo = bond.GetBondOrder()
        MolBond.append((a,b,bo))
        #if bond.IsInRing() or bond.IsAromatic():
        #    continue
        ChainBond.append(bond.GetIdx())
    return MolBond,ChainBond
#strings=  ['CCNNN',"CCC1CCCCC1","N=CCC1(=O)CCCC1"]
#getFingers(strings)
