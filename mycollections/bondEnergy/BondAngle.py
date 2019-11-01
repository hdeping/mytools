#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-09-01 15:27:40
    @project      : get bond and angles
    @version      : 1.0
    @source file  : DealBondEnergy.py

============================
"""

import numpy as np
import json
import sys
from mytools import MyCommon

class BondAngle(MyCommon):
    """
    docstring for BondAngle
    class: get bond and angles 
    name:
      atoms: str,list
      bonds:
           id: str,list
           pair: array (n)
           value: list,float
      angles:
           id: str,list
           pair: int,list ( 3 or 5)
           value: float,list
      dihedral:
           id: str,list
           pair: int,list (n,4)
           value: float,list
    """
    def __init__(self):
        super(BondAngle, self).__init__()
        
        self.filename = "residue.json"
    def getIdResidue(self):
        residues = self.loadJson(self.filename)
        # key: id
        # value: {"numbers","residues","energies","molecule"}
        results = {}

        # get atoms, bonds and angles
        parameters = self.loadJson("inchikey_filter_database.json")
        keys = ["atoms","bonds","angles"]
        for key in residues:
            # "[C](C(=O)C=O)Cl": {
            # "id": "AAIBHVXKVZWSCV-UHFFFAOYSA-N",
            # "molecule": "C(=O)C(=O)C(=O)Cl",
            # "energy": 166.9354999,
            # "number": 827}
    
            value = residues[key]
            id = value["id"]
            number = value["number"]
            energy = value["energy"]
            if id not in results:
                arr = {}
                arr["molecule"] = value["molecule"]
                arr["numbers"] = [number]
                arr["residues"] = [key]
                arr["energies"] = [energy] 
                for item in keys:
                    arr[item] = parameters[id][item]

                results[id] = arr
            else:
                results[id]["numbers"].append(number)
                results[id]["residues"].append(key)
                results[id]["energies"].append(energy)



        filename = "idResidue.json"
        self.writeJson(filename, results)


        
    def test(self):
        """
        test self.writeData and self.getStati
        """
        self.filename = "residue.csv"
        # id, energy and so on
        self.totalData = []
        self.filenames = self.getIDs()
        self.suffix = '_frag00.out'
        self.writeData()

        self.getStati()


    def writeData(self):
        """
        write csv to json
        """
        dicts = {}
        for line in self.totalData:
            result = {}
            key = line[4]
            energy = float(line[5])
            id = line[1]
            mol_smile = line[2]
            result["id"] = id 
            result["molecule"] = mol_smile
            result["energy"] = energy
            result["number"] = int(line[0])
            dicts[key] = result
        
        filename = self.filename.replace("csv","json")
        print("write data to ",filename)
        fp = open(filename,'w')
        json.dump(dicts,fp,indent=4)
        fp.close()
        
    def getIDs(self):
        """
        get IDs from the csv file
        """
        IDs = []
        # data = np.loadtxt(self.filename,delimiter=",",dtype=str)
        # read the data line by line
        fp = open(self.filename,'r')
        data = fp.read()
        data = data.split("\n")
        data.pop()
        fp.close()
        
        # second line is ID
        count = 0
        for line in data:
            count += 1
            line = line.split(",")
            self.totalData.append(line)
            id = line[1]
            if id not in IDs:
                IDs.append(id)
                self.stati[id] = 1 
            else:
                self.stati[id] += 1
        print("number of IDs",len(IDs))
        print("number of lines",count)
        
        return IDs

   
    def getStati(self):
        """
        get the static information of the molecules
        and residues
        """
        # print(self.stati)
        numbers = np.zeros(10,dtype=int)
        for key in self.stati:
            value = self.stati[key]
            numbers[value] += 1 
            if value == 4:
                print(key)
                
        print(numbers)
        # print(self.totalData)

    def getPara(self,ii):
        """
        get parameters from the file
        angles, lengths, and dihedral angles
        A,R and D
        input:
            ii, serial number
        return:
            bonds, length of bonds
            angle, bond angle
            dihedral, face-face angle
        """
        # location of the para file
        filename = "para/%s%s.para"%(self.filenames[ii],self.suffix)
        data = np.loadtxt(filename,delimiter=' ',dtype=str)

        numberR = 0
        numberA = 0
        # order: R,A,D
        for line in data[:,0]:
            first_letter = line[0]
            if first_letter == "R":
                numberR += 1
            elif first_letter == "A":
                numberA += 1

        # bonds,angles,dihedral
        bonds    = data[:numberR]
        angles   = data[numberR:numberR+numberA]
        dihedral = data[numberR+numberA:]

        return bonds,angles,dihedral
    def checkPara(self):
        """
        check if there is some broken lines
        """
        size = len(self.filenames)
        count = 0
        for i in range(size):
            data = getPara(i)
            # if the first letter is "R","A" or "D"
            letters = ["R","A","D"]
            #print(data[:,0])
            for line in data[:,0]:
                first_letter = line[0]
                #print(first_letter)
                if first_letter not in letters :
                    count = count + 1
                    print(i,first_letter)
                    break

            #break

        print("there are %d wrong samples"%(count))
        return
    def getSerial(self,ii):
        """
        get the atoms information from the file
        """
        filename = "serial/%s%s.serial"%(self.filenames[ii],self.suffix)
        data = np.loadtxt(filename,delimiter=' ',dtype=str)
        return data
    def arrToDicts(self,arr):
        """
        transform arrays into dictionaries
        """
        res = {}
        # arr to dicts

        for line in arr:
            key = line[1]
            value = float(line[2])
            res[key]  = value

        return res
    def getSample(self,ii):
        """
        get a sample information
        input:
            ii, 
        """
        result = {}
        # get atoms
        serial = self.getSerial(ii)
        result["atoms"] = list(serial[:,0])

        # get bonds, angles, dihedral
        bonds, angles, dihedral = self.getPara(ii)
        #print(bonds)
        #print(angles)
        #print(dihedral)

        result["bonds"] = self.arrToDicts(bonds)
        result["angles"] = self.arrToDicts(angles)
        result["dihedral"] = self.arrToDicts(dihedral)

        return result
    # write to the json file
    def writeToJson(self):
        size = len(self.filenames)
        #size = 100
        database = {}
        for i in range(size):
            print("sample",i+1)
            name = self.filenames[i]
            print(name)
            
            result = self.getSample(i)
            
            database[name] = result


        #### write the data
        filename = "inchikey_filter_database.json"
        self.writeJson(database,filename)

#bond = BondAngle()
#bond.getIdResidue()
# bond.writeToJson()
