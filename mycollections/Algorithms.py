#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-02-26 20:30:26
    @project      : practice for algorithms
    @version      : 0.1
    @source file  : Algorithms.py

============================
"""

import numpy as np

class Algorithms():
    """docstring for Algorithms"""
    def __init__(self):
        super(Algorithms, self).__init__()

    def possibleSites(self,arr,n = 8):
        """
        docstring for possibleSites
        arr:
            1d array
        return:
            array of possible sites
        """
        line = np.arange(n).tolist()
        if len(arr) == 0:
            return line
        else:
            num = len(arr)
            for index,i in enumerate(arr):
                sites = [i,i+num-index,i-num+index]
                for j in sites:
                    if j in line:
                        line.remove(j)
            return line
    
    def getChains(self,arr,n = 8):
        """
        docstring for getChains
        arr:
            2d array
        """
        total = []
        for line in arr:
            sites = self.possibleSites(line, n = n)
            res = []
            for i in sites:
                res.append(line + [i])
            total = total + res

        return total

    def eightQueens(self,n = 8):
        """
        docstring for eightQueens
        """
        res = [[]]
        for i in range(n):
            res = self.getChains(res,n = n)
        for i,line in enumerate(res):
            print(i,line)
        print("length: ",len(res))

        return
    def test(self):
        """
        docstring for test
        """
        self.eightQueens(n = 5)
        return
        
algo = Algorithms()
algo.test()
