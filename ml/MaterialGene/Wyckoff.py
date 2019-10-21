#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-09-25 16:52:21
    @project      : wyckoff site 
    @version      : 1.0
    @source file  : Wyckoff.py

============================
"""

import numpy as np

class Wyckoff():

    """Docstring for Wyckoff. """

    def __init__(self):
        """TODO: to be defined1. """

        super(Wyckoff,self).__init__()
        self.codeDir = "/home/hdeping/complexNetwork/00_CCodes/"
        self.dirs = self.codeDir + "24_CrystalFinderML/data/"
        filename = self.dirs + "lattice.csv"
        self.lattice = np.loadtxt(filename,
                                  delimiter=' ',
                                  dtype=int)

        self.wyckoff_site = [0,2, 6, 8, 20, 44, 
                             56, 60, 64, 66, 70, 72]

        self.wyckoff_type = ['a', 'e', 'b', 'i', 
                             'l', 'j', 'f', 'g', 
                             'd', 'h', 'c']

        return
    def getFullMatrix1(self):
        assert  len(wyckoff_seq) == 11
        fullMatrix = np.zeros((3,3,8))
        # deal with every site in the wyckoff sequences
        for i,site in enumerate(wyckoff_seq):
            # in the range of wyckoff_site (i,i+1)
            print(i,site)
            for j in range(wyckoff_site[i],wyckoff_site[i+1]):
                # get the lattice position
                ii,jj,kk = self.lattice[j]
                fullMatrix[ii,jj,kk] = int(site)
    
        return fullMatrix
    def test1(self):
        count_i = np.zeros(3)
        count_j = np.zeros(3)
        count_k = np.zeros(8)
        for i,j,k in self.lattice:
            count_i[i] += 1
            count_j[j] += 1
            count_k[k] += 1
        
        print(count_i)
        print(count_j)
        print(count_k)
        
        fullMatrix = getFullMatrix(wyckoff_seq)
        print(fullMatrix)
        
        print(fullMatrix.shape)
        return



    def getFullMatrix(self):
        filename = self.dirs + "01_3-3-8_194_energy.csv"
        data   = np.loadtxt(filename,delimiter=',',dtype=float)
    
        wyckoff_seq = data[:,:-1].astype(int)
        labels = data[:,-1]
    
        length = len(wyckoff_seq[0])
        assert  length == 11
        row = len(wyckoff_seq)
        fullMatrix = np.zeros((row,3,3,8))
        #print(wyckoff_seq.shape)
        # deal with every site in the wyckoff sequences
    
        for i in range(length):
            # in the range of wyckoff_site (i,i+1)
            site = wyckoff_seq[:,i]
            #print(i,len(site))
    
            for j in range(wyckoff_site[i],wyckoff_site[i+1]):
                # get the lattice position
                ii,jj,kk = lattice[j]
                fullMatrix[:,ii,jj,kk] = site
    
        return fullMatrix,labels
