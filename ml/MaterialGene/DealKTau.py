#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-09-25 18:24:27
    @project      : Material Genome Prediction
    @version      : 1.0
    @source file  : DealKTau.py

============================
"""

import numpy as np
import sys
import os
# k-tau
from scipy.stats import kendalltau
    
class DealKTau(object):

    """Docstring for DealKTau. """

    def __init__(self):
        """TODO: to be defined1. """
        super(DealKTau,self).__init__()
    def getKtau(self,number):
        """
        read data from "test_result%d.txt"%(number)
        and get k-tau
        definition of k-tau:
        input:
            number, serial number
        return:
            k_tau, error
        """
        filename = "test_result%d.txt"%(number)
        data = np.loadtxt(filename,delimiter=',',dtype=float)
        n = len(data)
        #print(n)
        res = 0
        n1 = 0
        # real energy
        x = data[:,1]
        # predicted energy
        y = data[:,0]
        k_tau = kendalltau(x,y)[0]
        error = np.linalg.norm(x-y)/np.sqrt(len(x))
        return k_tau,error
    def test(self):
        """
        test for self.getKtau
        """
        fp = open("k_tau.txt",'w')
        for i in range(5,101,5):
            k_tau,error = self.getKtau(i)
            print(i,k_tau,error)
            fp.write("%d,%f,%f\n"%(i,k_tau,error))
        fp.close()
