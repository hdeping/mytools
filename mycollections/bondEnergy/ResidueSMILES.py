#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-09-01 16:44:16
    @project      : get the smiles format of the residue
    @version      : 0.1
    @source file  : ResidueSMILES.py

============================
"""

import openbabel as ob
import pybel as pb
import os
import numpy as np
import json
from DealBondEnergy import BondAngle

class ResidueSMILES(BondAngle):
    """docstring for ResidueSMILES"""
    def __init__(self):
        super(ResidueSMILES, self).__init__()
        
    # Break the bond in ring and generate the radical fragment
    def BreakRing(self,mol):
        ring_frag = ob.OBMol()
        for atom in ob.OBMolAtomIter(mol):
            ring_frag.AddAtom(atom)
        for bond in ob.OBMolBondIter(mol):
            ring_frag.AddBond(bond)
        return ring_frag

    # Determinate the second-end atom
    def IsNearTerminal(self,atom):
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
    def FillAtom(self,mol, atom, frag, fatom_idx):
        # mol, frag -- Class OBMol
        # atom -- Class OBAtom
        # fatom_idx -- Record the atom has existed
        frag.AddAtom(atom)
        fatom_idx.append(atom.GetIdx())
        if atom.GetValence == 0:
            return frag
        elif self.IsNearTerminal(atom):
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
                    self.FillAtom(mol, _atom, frag, fatom_idx)

    # BondLink() is a function to bonding the atoms in the fragment 
    def FragBondLink(self,frag, fatom_idx,mol_bond,NumAtomsNoH):
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
    def SimplifyLs(self,ls1, ls2):
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


    # input: string of SMILES format
    # output: the SMILES format of the residue
    def getResidues(self,smi_string):
        
        #file's type conversion and generate a 3D builder
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("smi", "smi")

        # create a molecule container
        mol = ob.OBMol()
        obConversion.ReadString(mol, smi_string)
        NumAtomsNoH = mol.NumAtoms()
        NumOfBonds = mol.NumBonds()
        #print("atom numbers",NumAtomsNoH)
        MolSmi = pb.Molecule(mol).write("smi").replace('\t\n','')
        #mol.AddHydrogens()

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
        #MolBond.sort()
        NumOfRingBond = len(MolBond) - len(ChainBond)

        # prepare to store fragment's SMILES
        ListOfFrag1 = []
        ListOfFrag2 = []
        atomId = []
        for BondIdx in range(len(MolBond)):
            obConversion.ReadString(mol, smi_string)
            #mol.AddHydrogens()
            a,b,bo = MolBond[BondIdx]
            # if A or B is oxygen
            AisO = mol.GetAtom(a).IsOxygen()
            BisO = mol.GetAtom(b).IsOxygen()
            AisC = mol.GetAtom(a).IsCarbon()
            BisC = mol.GetAtom(b).IsCarbon()
            #if (AisO or BisO) == False:
            #    print(a,b,False)
            #    continue
            #else:
            #    print(a,b,True)
            if (AisO or BisO) == False:
                continue
            if (AisC or BisC) == False:
                continue
            

            Frag1 = ob.OBMol()
            Frag1_idx = []
            Frag2 = ob.OBMol()
            Frag2_idx = []

            # find the begin atom and the end atom of the 
            # breaking bond. 
            a1 = mol.GetBond(BondIdx).GetBeginAtom()
            a2 = mol.GetBond(BondIdx).GetEndAtom()
            breakBO = mol.GetBond(BondIdx).GetBondOrder()
            # bonds information
            bond = mol.GetBond(BondIdx)


            # homolysis, create radical 
            a1.SetSpinMultiplicity(breakBO+1)
            a2.SetSpinMultiplicity(breakBO+1)
            mol.DeleteBond(mol.GetBond(BondIdx))

            if BondIdx in ChainBond:
                self.FillAtom(mol, a1, Frag1, Frag1_idx)
                self.FillAtom(mol, a2, Frag2, Frag2_idx)

                self.FragBondLink(Frag1, Frag1_idx, MolBond, NumAtomsNoH)
                self.FragBondLink(Frag2, Frag2_idx, MolBond, NumAtomsNoH)

                frag1_smi = pb.Molecule(Frag1).write("smi").replace('\t\n','')
                frag2_smi = pb.Molecule(Frag2).write("smi").replace('\t\n','')
            else:
                Frag1 = self.BreakRing(mol)
                frag2_smi = pb.Molecule(Frag1).write("smi").replace('\t\n','')
                frag1_smi = ''

            if frag2_smi == "[OH]" or frag2_smi == "[O]":
                ListOfFrag1.append(frag1_smi)
                ListOfFrag2.append(frag2_smi)
                # get bonds
                atomId.append(MolBond[BondIdx])
        #ListOfFrag = SimplifyLs(ListOfFrag1,ListOfFrag2)
        #DelDuplFrag(ListOfFrag)
        #print("list 1")
        #print(ListOfFrag1)
        #print(ListOfFrag1,ListOfFrag2)
        #print("list 2")
        #print(ListOfFrag2)
        #print("bonds")
        #print(atomId)
        #get_frag_tbl(MolSmi,label, ListOfFrag)
        # Output Gaussian input file
        return atomId,ListOfFrag1
    def getSMILES(self,filename):
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

    def getMolecules(self):
        filename = "idResidue.json"
        data = self.loadJson(filename)
        molecules = []
        residues = []
        self.resToID = {}
        for key in data:
            molecule = data[key]["molecule"]
            molecules.append(molecule)
            for residue in data[key]["residues"]:
                residues.append(residue)
                self.resToID[residue] = key

        self.totalData = data
        return residues,molecules


    def test(self):
        residues,molecules = self.getMolecules()
        print(self.resToID)
        #print(self.resToID)
        

    # run the main program
    def run(self):
        # get molecules
        self.resToID = None
        self.totalData = None
        residues,molecules = self.getMolecules()
        count = np.zeros(10)

        total = 0
        freq = 0
        filename = "output.result"
        fp = open(filename,'w')

        mismatch = 0
        match    = 0
        residueAtoms = {}
        for i,smi_string in enumerate(molecules):
            print("################# %d compounds #########"%(i))
            fp.write("################# %d compounds #########"%(i)+'\n')
            fp.write(smi_string + '\n')
            atomId, frag = self.getResidues(smi_string)
            for ii,line in enumerate(frag):
                if line in residues:
                    fp.write(line + '\n')
                    match += 1
                    res = {}
                    res["id"] = self.resToID[line]
                    res["target"] = atomId[ii][:2]
                    residueAtoms[line] = res
                    print(atomId[ii])
                else:
                    print()
                    fp.write(line + " is not in data"+'\n')
                    mismatch += 1
                
            #total += num
            #count[num] += 1
            #freq += 1
            #if freq == 20000:
            #    break
            #if i == 100:
            #    break

        fp.close()
        #count = count.astype(int)
        #print("freq",freq)
        #print("count",count)
        #print("total",total)
        #print("mismatch:",mismatch)
        #print("match:",match)
        self.writeJson("residueAtoms.json",residueAtoms)

residue = ResidueSMILES()
residue.run()
#residue.test()
