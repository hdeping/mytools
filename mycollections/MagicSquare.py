#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-09-30 10:05:02
    @project      : Magic Square of odd order
    @version      : 1.0
    @source file  : MagicSquare.py

============================
"""

import numpy as np
import pandas 
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Process
from tqdm import tqdm

class MagicSquare():
    """docstring for MagicSquare"""
    def __init__(self):
        super(MagicSquare, self).__init__()
    
    def setOrder(self,n):
        if n%2 == 1:
            print("set order to ",n)
            self.order = n 
        else:
            print("%d is not a odd number"%(n))
            print("please input an odd number such as 3,5...")
            self.order = 0
            
        return        
    # periodic site
    def newSite(self,i):
        if i < 0:
            i += self.order
        if i >= self.order:
            i -= self.order
        return i
    def move(self,magic_square,i,j):
        ii = i - 1
        jj = j + 1
        ii = self.newSite(ii)
        jj = self.newSite(jj)

        # if site (ii,jj) is ocupied
        if magic_square[ii,jj] != 0:
            # next line of the site (i,j)
            #print("(%d,%d) is ocupied"%(ii,jj))
            ii = i + 1
            ii = self.newSite(ii)
            jj = j
            
            
        return ii,jj
    def judgeMagic(self,magic_square):
        """
        magic_square:
            2d array of n*n
        return:
            yes or no
        """

        n = len(magic_square)
        magic_square = np.array(magic_square)
        total = n*(n**2 + 1) // 2

        # n rows, n columns, tow diagonals
        arr_sum = np.zeros(2*n+2)
        arr_sum[:n] = np.sum(magic_square,axis=0)
        arr_sum[n:2*n] = np.sum(magic_square,axis=1)
        for i in range(n):
            arr_sum[2*n] += magic_square[i,i]
            arr_sum[2*n+1] += magic_square[i,n - 1 - i]
        judge = (arr_sum == total)
        print(arr_sum)
        if sum(judge) == 2*n+2:
            print("It is a %dth order magic square"%(n))
        else:
            print("It is not a magic square")

        magic_square = pandas.DataFrame(magic_square)
        magic_square.plot.bar(stacked=True) 
        plt.show()

        return 
    # get a magic square with horse-step method  
    def magicSquare(self):
        n = self.order
        magic_square = np.zeros((n,n),int)

        i = 0
        j = n // 2
        magic_square[i,j] = 1
        num = n*n - 1
        for k in range(num):
            #print(i,j)
            #print(magic_square)
            i,j = self.move(magic_square,i,j)
            magic_square[i,j] = k+2

        print(magic_square)
        self.judgeMagic(magic_square)

        return
    # the second method
    def initMatrix(self):
        n = self.order
        magic_square = np.zeros((2*n-1,2*n-1),int)
        # initialize the matrix
        for i in range(n):
            for j in range(n):
                ii = i+j
                jj = n-1-i+j
                magic_square[ii,jj] = n*i + j + 1
        # print(magic_square)

        return magic_square

    def adjustMatrix(self,magic_square):
        n = self.order
        m = n // 2
        for i in range(m):
            for j in range(i+1):
                # first m rows and rows
                ii = i
                jj = n-1-i+2*j
                # symmetric sites
                # print(ii,n,jj)
                magic_square[ii+n,jj] = magic_square[ii,jj]
                magic_square[jj,ii+n] = magic_square[jj,ii]
                
                # first m rows
                ii = 2*n - 2 - ii
                jj = n-1-i+2*j
                # previous line of symmetric sites
                magic_square[ii-n,jj] = magic_square[ii,jj]
                magic_square[jj,ii-n] = magic_square[jj,ii]
        magic_square = magic_square[m:m+n,m:m+n]
        return magic_square

    def magicSquare1(self):
        magic_square = self.initMatrix()
                
        # adjust the matrix
        magic_square = self.adjustMatrix(magic_square)
        self.judgeMagic(magic_square)
            
        print(magic_square)

        return

    def isListRepeated(self,arr1,arr2):
        """
        docstring for isListRepeated
        """
        for i in arr1:
            if i in arr2:
                return True 
        return False

    def getNonRepeatCombinations(self,arr):
        """
        docstring for getNonRepeatCombinations
        arr:
            2d array with the size (N,n)
        """
        combinations2 = itertools.combinations(arr,2)

        total2 = []
        for line in combinations2:
            if not self.isListRepeated(line[0],line[1]):
                total2.append(line[0]+line[1])

        return total2
    def getProductCombinations(self,line,n = 4):
        """
        docstring for getProductCombinations
        line:
            1d array, n*n items
        """
        indeces = itertools.product(np.arange(n),repeat=n)
        combinations = []
        for index in indeces:
            res = []
            for i in range(n):
                begin = n*i
                res.append(line[begin+index[i]])
            combinations.append(res)
        return combinations


    def checkDiagonal(self,items):
        """
        docstring for checkDiagonal
        items:
            array of (n,n)
        """
        num = [0,0]
        for i in range(self.n):
            num[0] += items[i,i]
            num[1] += items[i,self.n - i - 1]
        if num == [self.Sum,self.Sum]:
            return True 
        else:
            return False

    def arrangeSquare(self,arr):
        """
        docstring for arrangeSquare
        arr:
            2d array of (N,n*n)
        """

        results = []

        for line in tqdm(arr):
            items = self.getProductCombinations(line)
            items = self.getPossibleCombinations(items)
            res = []
            for item in items:
                item = np.array(item).reshape((self.n,self.n))
                if self.checkDiagonal(item):
                    res.append(item)
            results += res


        return results

    def getPossibleCombinations(self,combinations):
        """
        docstring for getPossibleCombinations
        """

        totalSum = []
        for line in combinations:
            if sum(line) == self.Sum:
                totalSum.append(list(line))

        # get all the non-repeated combinations
        for i in range(2):
            totalSum = self.getNonRepeatCombinations(totalSum)

        return totalSum 

    def rotateMatrix(self,matrix):
        """

        docstring for rotateMatrix
        rotate the matrix with pi/2
        00 01 ==> 10 00
        10 11     11 01
        00 01 02 03 ==> 30 20 10 00 
        10 11 12 13     31 21 11 01
        20 21 22 23     32 22 12 02
        30 31 32 33     33 23 13 03 

        """
        res = np.ones(matrix.shape,int)
        n = matrix.shape[0]
        for i in range(n):
            res[:,i] = matrix[n-1-i,:]
        
        return res
    def testRotateMatrix(self):
        """
        docstring for testRotateMatrix
        """
        a = np.array([[1,2,3,4],[1,12,3,4],
                      [1,2,3,34],[1,2,23,4]])
        print(a)
        for i in range(3):
            a = self.rotateMatrix(a)
            print(i)
            print(a)
            print(a.transpose())
        return

    def matrix2List(self,matrix,inv=False):
        """
        docstring for matrix2List
        matrix:
            n*n, 2d array when inv = False
            list of length n when inv = True
        """
        if inv:
            res = np.array(matrix)
            res = res.reshape((self.n,-1))
        else:
            res = matrix.reshape(-1).tolist()
            
        return res 

    def rotationExpansion(self,results):
        """
        docstring for rotationExpansion
        arr:
            3d array of (N,n,n)
        rotate a matrix with pi/2, pi and 3pi/2
        and their transpose

        """

        res = []

        for i,line in enumerate(results):
            items = [] 
            arr = self.matrix2List(line)
            items.append(arr)
            arr = self.matrix2List(line.transpose())
            items.append(arr)

            for j in range(3):
                line = self.rotateMatrix(line)
                arr = self.matrix2List(line)
                items.append(arr)
                arr = self.matrix2List(line.transpose())
                items.append(arr)
            for item in items:
                if item not in res:
                    res.append(item)

        return res
    def getAllSquareFour(self):
        """
        docstring for getAllSquareFour
        1 2 3 4
        5 6 7 8
        9 10 11 12
        13 14 15 16

        the sum of the middle four
        """
        n      = 4
        self.n = n
        self.Sum = n*(n*n+1)//2
        arr = np.arange(1,n*n+1)
        combinations = itertools.combinations(arr,n)

        totalSum = self.getPossibleCombinations(combinations)

        print(len(totalSum))
        # print(totalSum[:2])
        
        begin = 0 
        end   = len(totalSum)
        results = self.arrangeSquare(totalSum[begin:end])

        results = self.rotationExpansion(results)
        # print(len(results))
        for i,line in enumerate(results):
            print(i,line)

        return

    def test(self):
        """
        docstring for test
        """
        self.getAllSquareFour()
        # self.testRotateMatrix()

        return

magic = MagicSquare()
# magic.setOrder(7)
# magic.magicSquare1()
magic.test()