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
from Formulas import Formulas

class Algorithms(Formulas):
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

    def testQueens(self):
        """
        docstring for testQueens
        4 2     10 724
        5 10    11 2680
        6 4     12 14200
        7 40    13 73712
        8 92    14 365596
        9 352   15 2279184
                16 14772512 
        """

        t1 = time.time()
        # res = self.eightQueens(n = n)
        # t2 = time.time()
        # print(len(res),"time = ",t2 - t1)
        # t1 = t2 
        for n in range(4,15):
            
            self.count = 0
            self.queens([0]*n,0)
            t2 = time.time()
            print(n,self.count,"time = ",t2 - t1)
            t1 = t2

        return 

    def polynomialMulti(self,arr1,arr2):
        """
        docstring for polynomialMulti
        arr1,arr2:
            1d array, coefficients of the polynomial
        """
        num1 = len(arr1)
        num2 = len(arr2)
        if num1 > num2:
            arr1,arr2 = arr2,arr1
            num1,num2 = num2,num1 
        arr1 = np.array(arr1)
        res = np.zeros(num1+num2-1,int)

        for index,i in enumerate(arr2):
            res[index:index+num1] += arr1*i

        return res

    def polynomialPow(self,arr,n):
        """
        docstring for polynomialPow
        """
        if n == 1:
            return arr 
        else:
            res = arr 
            for i in range(n-1):
                res = self.polynomialMulti(res,arr)
            return res
        
    def generateFunc(self):
        """
        docstring for generateFunc
        generating functions in combinatorics
        (1+x+x^2+...)(1+x+x^2...)

        for x1 + x2 + .. x5 = 100
        one can compute (ax+...x^96)^5
        C(99,4) ==> 38225
        permutation and combination
        """
        n = 100 
        m = 5
        arr = [1,1,1,1]*(n - m + 1)
        arr[0] = 1
        res = self.polynomialPow(arr,m)
        print(res[100])
        
        return
    def test(self):
        """
        docstring for test
        """
        # self.testQueens()
        self.generateFunc()
        # print(self.getCombinatorEqnSolNumByIter(5,95))
        return
        
algo = Algorithms()
algo.test()
