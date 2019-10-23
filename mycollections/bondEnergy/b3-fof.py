#coding=utf-8
import openbabel as ob
import pybel as pb
from optparse import OptionParser
import os

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


# Simplify is a function to remove the same fragment pair
def SimplifyLs(ls1, ls2):
    ls = zip(ls1,ls2)
    for i in range(len(ls)):
        a,b = ls[i]
        if len(a)<=len(b):
            continue
        else:
            ls[i] = (b,a)
    ls.sort()
    i = 0
    while i < len(ls)-1 :
        if ls[i] == ls[i+1]:
            ls.remove(ls[i])
        else:
            i += 1
    return ls


# remove the duplicate fragments
def DelDuplFrag(frag_list):
        Frag_inchi = []
        i = 0
        while i < len(frag_list):
                try:
                        x_inchi = pb.readstring("smi", frag_list[i][0]).write("inchi")
                except:
                        x_inchi = " "
                y_inchi = pb.readstring("smi", frag_list[i][1]).write("inchi")
                if (x_inchi, y_inchi) in Frag_inchi:
                        frag_list.pop(i)
                else:
                        Frag_inchi.append((x_inchi,y_inchi))
                        i = i+1


def get_frag_tbl(MolInchi,filename, frag_list):
    with open(filename + '_FragList' +'.txt', 'a') as file:
        for i in range(len(frag_list)):
            file.write(MolInchi+"\t"+frag_list[i][0]+"\t"+frag_list[i][1]+"\n")


def main():
    # set commandline mode
    # specify the molecule file from the command-line 
    optparser = OptionParser(usage="BDEdb [-i|--input <filename>]")
    optparser.add_option("-i", "--input", action="store", type="string", dest="filename", help='''input molecule file with openbabel accepted formats and the process can generator the data about each bond's BDE''')
    options, args = optparser.parse_args()
    filename = options.filename
    label, inp_flt = os.path.splitext(filename)
    print("Making fragments for the molecule in {}\n".format(filename))
    filetype = inp_flt[1:]
    
    #file's type conversion and generate a 3D builder
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats(filetype, "gjf")

    # create a molecule container
    mol = ob.OBMol()
    obConversion.ReadFile(mol, filename)
    NumAtomsNoH = mol.NumAtoms()
    NumOfBonds = mol.NumBonds()
    MolSmi = pb.Molecule(mol).write("smi").replace('\t\n','')
    mol.AddHydrogens()

    # get the bond information
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
    MolBond.sort()
    NumOfRingBond = len(MolBond) - len(ChainBond)

    # prepare to store fragment's SMILES
    ListOfFrag1 = []
    ListOfFrag2 = []
    for BondIdx in range(len(MolBond)):
        obConversion.ReadFile(mol, filename)
        mol.AddHydrogens()

        Frag1 = ob.OBMol()
        Frag1_idx = []
        Frag2 = ob.OBMol()
        Frag2_idx = []

        # find the begin atom and the end atom of the 
        # breaking bond. 
        a1 = mol.GetBond(BondIdx).GetBeginAtom()
        a2 = mol.GetBond(BondIdx).GetEndAtom()
        breakBO = mol.GetBond(BondIdx).GetBondOrder()

        # homolysis, create radical 
        a1.SetSpinMultiplicity(breakBO+1)
        a2.SetSpinMultiplicity(breakBO+1)
        mol.DeleteBond(mol.GetBond(BondIdx))

        if BondIdx in ChainBond:
            FillAtom(mol, a1, Frag1, Frag1_idx)
            FillAtom(mol, a2, Frag2, Frag2_idx)

            FragBondLink(Frag1, Frag1_idx, MolBond, NumAtomsNoH)
            FragBondLink(Frag2, Frag2_idx, MolBond, NumAtomsNoH)

            frag1_smi = pb.Molecule(Frag1).write("smi").replace('\t\n','')
            frag2_smi = pb.Molecule(Frag2).write("smi").replace('\t\n','')
        else:
            Frag1 = BreakRing(mol)
            frag2_smi = pb.Molecule(Frag1).write("smi").replace('\t\n','')
            frag1_smi = ''

        ListOfFrag1.append(frag1_smi)
        ListOfFrag2.append(frag2_smi)
    ListOfFrag = SimplifyLs(ListOfFrag1,ListOfFrag2)
    DelDuplFrag(ListOfFrag)
    print(ListOfFrag)
    #get_frag_tbl(MolSmi,label, ListOfFrag)
    # Output Gaussian input file
    Frags = [ element for item in ListOfFrag for element in item ]
    print(Frags)
    Frags_check = []
    Smi2gjf(MolSmi)
    AdGJF(MolSmi, "frag00")    
    for i in range(len(Frags)):
        frag = Frags[i]
        if frag in Frags_check:
            continue
        else:
            if IsInDB(frag) == False:
                Smi2gjf(frag)
                AdGJF(frag, "frag" + str(i))
                Frags_check.append(frag)
    
    FragInfo_filler()
    for i in range(len(ListOfFrag)):
        frag1 = ListOfFrag[i][0]
        frag2 = ListOfFrag[i][1]
        dbFiller(MolSmi, frag1, frag2)    

        os.remove('info.txt')
        os.remove('mol.smi')

main()

