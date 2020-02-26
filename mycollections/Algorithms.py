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
import time

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

    def printQueens(self,line):
        """
        docstring for printQueens
        """
        n = len(line)
        for i in line:
            string = ["0"]*n 
            string[i] = "1"
            string = " ".join(string)
            print(string)
        return

    def eightQueens(self,n = 8):
        """
        docstring for eightQueens
        """
        res = [[]]
        for i in range(n):
            print("i = ",i)
            res = self.getChains(res,n = n)
        # for i,line in enumerate(res):
        #     print(i)
        #     self.printQueens(line)
        # print("length: ",len(res))

        return res

    def printDiff(self,arr):
        """
        docstring for printDiff
        """
        # print(arr)
        arr = np.array(arr)
        line = [arr[0]]
        length = len(arr)
        for i in range(1,length):
            k = arr[i] - arr[i-1]
            if k < 0:
                k = k + length 
            line.append(k)
        print(self.count,line)
        # print(self.count)
        # self.printQueens(arr)

        return
    def queens(self,arr,length):
        """
        docstring for queens
        get the queens by recursion
        """
        if length == len(arr):
            self.count += 1
            # self.printDiff(arr)
            return
        else:
            for i in range(len(arr)):
                arr[length] = i
                judge = 1
                for j in range(length):
                    if arr[j] == i or abs(arr[j] - i) == length - j:
                        judge = False 
                        break
                if judge:
                    self.queens(arr,length+1)

        return

    def test(self):
        """
        docstring for test
        """
        n = 5
        t1 = time.time()
        # res = self.eightQueens(n = n)
        # t2 = time.time()
        # print(len(res),"time = ",t2 - t1)
        # t1 = t2 

        self.count = 0
        self.queens([0]*n,0)
        t2 = time.time()
        print(self.count,"time = ",t2 - t1)
        return
        
algo = Algorithms()
algo.test()
