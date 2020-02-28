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
from sympy import *
from Formulas import Formulas

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
        # print(arr_sum)

        # magic_square = pandas.DataFrame(magic_square)
        # magic_square.plot.bar(stacked=True) 
        # plt.show()
        if sum(judge) == 2*n+2:
            print("It is a %dth order magic square"%(n))
            return True
        else:
            print("It is not a magic square")
            return False


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
        a = np.arange(1,17).reshape((4,4))
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

    def getCenters(self,results):
        """
        docstring for getCenters
        """
        count = 0

        centers = {}
        for i,line in enumerate(results):
            # print(i)
            line = self.matrix2List(line,inv=True)
            # count += self.judgeMagic(line)
            line = self.matrix2List(line[1:3,1:3])
            res = line.copy()
            line.sort()
            line = str(line)
            if line in centers:
                centers[line].append(res)
            else:
                centers[line] = [res]
            # print(line)

        print(len(results),count)


        for key in centers:
            values = centers[key]
            print(key,len(values),values)
        print(len(centers))
        print(centers)

        return
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

        # results = self.rotationExpansion(results)
        # self.getCenters(results)


        return

    def getHexaSegments(self,lengths):
        """
        docstring for getHexaSegments
        lengths:
            1d array, [m,m+1,...,2m-1,2m-2...,m]
        """
        segments = []
        count = 0 
        index = 0
        res   = []
        for i in range(self.n):
            if count < lengths[index]:
                res.append(i)
                count += 1 
            else:
                segments.append(res)
                count = 1
                res = [i]
                index += 1 
        segments.append(res)
        return segments

    def getHexaEqn(self):
        """
        docstring for getHexaEqn
        get all the equations to satify 
        the magic definition
        """

        row = 2*self.m - 1

        
        lengths = []
        for i in range(self.m - 1):
            lengths.append(self.m+i)
        lengths.append(row)
        for i in range(self.m - 1):
            lengths.append(lengths[self.m - 2 - i])

        print(lengths)

        # get the segments
        segments = self.getHexaSegments(lengths)

        matrix = - np.ones((2,row,row),int)
        for i in range(self.m-1):
            j = -i - 1
            matrix[0,i,:lengths[i]] = segments[i]
            matrix[0,j,-lengths[i]:] = segments[j]

            matrix[1,i,-lengths[i]:] = segments[i]
            matrix[1,j,:lengths[i]] = segments[j]
        for i in range(2):
            matrix[i,self.m-1,:] = segments[self.m-1]

        print(matrix)
        equations = []

        x = symbols("x0:%d"%(self.n))
        solNum = 0
        y = symbols("y0:%d"%(solNum))
        self.eqnIndeces = []
        for i in range(row):
            indeces = [[],[]]
            for j in range(row):
                k = matrix[0,i,j]
                if k >= 0:
                    indeces[0].append(k)
                k = matrix[0,j,i]
                if k >= 0:
                    indeces[1].append(k)
            self.eqnIndeces += indeces
        for i in range(row):
            indeces = []
            for j in range(row):
                k = matrix[1,j,i]
                if k >= 0:
                    indeces.append(k)
            self.eqnIndeces.append(indeces)

        for line in self.eqnIndeces:
            res = 0
            for i in line:
                if i < solNum:
                    res += y[i]
                else:
                    res += x[i]
            res = res - self.Sum
            equations.append(res)
        equations.remove(equations[2])
        equations.remove(equations[0])
        # print(equations,len(equations))
        
        # answers = solve(equations[:-1],x)
        # for i,key in enumerate(answers):
        #     print(i,key,answers[key])

        return

    def checkMagicHexa(self,x):
        """
        docstring for checkMagicHexa
        x:
            array of length self.n
               
        x[10] =  -x[13]  - x[14]  - x[15]  - x[17]  + x[4]  + 38
        x[9]  = x[13]  + x[15]  + x[17]  + x[3]  - x[4]  - 38
        x[8]  = -x[13]  - x[17]  - x[3]  + 38
        x[11] =  -x[15]  - x[18]  + 38
        x[16] =  -x[17]  - x[18]  + 38
        x[12] =  -x[13]  - x[14]  - x[15]  + 38
        x[7]  = x[13]  + x[14]  + x[15]  + x[17]  + x[18]  - 38
        x[6]  = x[13]  + x[15]  - x[4] 
        x[5]  = -x[13]  - x[15]  - x[3]  + 38
        x[2]  = -x[13]  + x[18]  + x[4] 
        x[1]  = 2*x[13]  + x[14]  + x[15]  + x[17]  + x[3]  - x[4]  - 38
        x[0]  = -x[13]  - x[14]  - x[15]  - x[17]  - x[18]  - x[3]  + 76
        """

        for line in self.eqnIndeces:
            res = 0
            for i in line:
                res += x[i]
            if res != self.Sum:
                print("It is not a magic hexagon")
                return False 
        print("YES! It is a magic hexagon",x)
        return True

    def testHexaPerm(self):
        """
        docstring for testHexaPerm
        """
        
        total = []
        for i in range(self.m):
            total.append([])
        for i in range(self.m-2):    
            combinations = itertools.combinations(arr,self.m+i)
            count = 0
            for line in combinations:
                count += 1
                if sum(line) == self.Sum:
                    total[i].append(list(line))
            print(len(total[i]),count)

        count = len(total[0])
        print(total[0])
        total = total[0]
        matrix = np.zeros((count,count),int)
        
        for i in range(count):
            for j in range(i+1,count):
                print(i,j)
                matrix[i,j] = self.isTupleAdjacent(total[i],total[j])
                matrix[j,i] = matrix[i,j]

        permutations = itertools.permutations(arr,2)
        count = 0
        total = []
        for line in permutations:
            count += 1
            if sum(line) >= self.n:
                total.append(list(line))
        print(len(total),count)

        permutations = itertools.permutations(total,3)
        count = 0
        total2 = []
        for line in tqdm(permutations):
            count += 1
            p = 1
            for i in range(3):
                p1 = (line[i-1][1] + line[i][0] >= self.n)
                p = p*p1
            if p:
                total2.append(list(line))
                print(line)
            if count == 100:
                break
        print(len(total2),count)

        return

    def isTupleAdjacent(self,arr1,arr2):
        """
        docstring for isTupleAdjacent
        """
        for i in arr1:
            for j in arr2:
                if i == j:
                    return True
        return False

    def checkHexa(self,line):
        """
        docstring for checkHexa
        line:
            1d array with length 6
        check if the input could produce a magic hexagon
        """
        x = [0]*self.n
        indeces = [0,2,11,18,16,7]
        for index,i in enumerate(indeces):
            x[i] = line[index]

        x[1]  = self.Sum - x[0]  - x[2] 
        x[6]  = self.Sum - x[2]  - x[11]
        x[15] = self.Sum - x[11] - x[18] 
        x[17] = self.Sum - x[18] - x[16]
        x[12] = self.Sum - x[16] - x[7] 
        x[3]  = self.Sum - x[7]  - x[0]
        c1    = self.Sum - x[3] - x[6]
        c2    = self.Sum - x[1] - x[15]
        c3    = self.Sum - x[6] - x[17]
        c4    = self.Sum - x[12] - x[15]
        c5    = self.Sum - x[3] - x[17]
        c6    = self.Sum - x[1] - x[12]
        X4 = []
        for i in range(1,self.n+1):
            if i not in x:
                X4.append(i)

        # print(x,X4)
        for i in X4:
            x[4]  = i
            x[5]  = c1 - x[4]
            x[10] = c2 - c1 + x[4]
            x[14] = c3 - c2 + c1 - x[4]
            x[13] = c4 - c3 + c2 - c1 + x[4]
            x[8]  = c5 - c4 + c3 - c2 + c1 - x[4]
            x[9]  = self.Sum - x[7] - x[8] - x[10] - x[11]
            count = 0
            for j in [5,10,14,13,8,9]:
                if x[j] <= 0:
                    count = 1
                    break
            if count == 1:
                break
            arr = x.copy()
            arr.sort()
            count = 0
            for i in range(len(arr)-1):
                if arr[i] == arr[i+1]:
                    count += 1 
                    break
            # print(x)
            if count == 0:
                print(x)

        return

    def magicHexa(self):
        """
        docstring for magicHexa
             0   1   2
           3   4   5   6
         7   8   9   10   11
           12  13  14   15
             16  17  18 
        two wrong cases:
           14, 19, 5
          9, 7, 6, 16
        15, 1, 2, 10, 17
          11, 20, 4, 3
            12, 8, 18

            15, 11, 12
          10,  1, 20, 7
        13,  17, 4, 2, 19
           9, 3,  21, 5
             16, 8, 14
        right cases
            10,13,15
          12, 4, 8, 14
        16, 2, 5, 6,  9
          19, 7, 1, 11
            3, 17,18
        6*38 = 2*(19*10 - x)
        x = 

        """
        m = 3
        self.m = m
        if self.m == 1:
            self.n = 1 
        else:
            self.n = 1+3*m*(m-1)
        self.Sum = self.n*(self.n+1)//(4*m-2)
        print(self.m,self.n,self.Sum)

        arr = np.arange(1,self.n+1)
        permutations = itertools.permutations(arr,6)

        self.checkHexa([19,18,16,15,14,12])
        count = 0
        for line in tqdm(permutations):
            p = 1 
            for i in range(6):
                p1 = (line[i-1] + line[i] >= 19)
                p  = p*p1
            if p:
                # print(line)
                self.checkHexa(line)
            count += 1 
            # if count % 1000000 == 0:
            #     print(count)
            #     break
            break

        print(count)



        return
    def testMagicHexa(self):
        """
        docstring for testMagicHexa
        """

        self.magicHexa()
        self.getHexaEqn()
        arr = [14, 19, 5, 9, 7, 6, 16,15, 1, 2, 10, 17,11, 20, 4, 3,12, 8, 18]
        self.checkMagicHexa(arr)
        arr = [15, 11, 12,10, 1, 20, 7,13, 17, 4, 2, 19,9, 3, 21, 5,16, 8, 14]
        self.checkMagicHexa(arr)
        arr = [[10, 13, 15, 12, 4, 8, 14, 16, 2, 5, 6, 9, 19, 7, 1, 11, 3, 17, 18],
               [10, 12, 16, 13, 4, 2, 19, 15, 8, 5, 7, 3, 14, 6, 1, 17, 9, 11, 18],
               [18, 17, 3, 11, 1, 7, 19, 9, 6, 5, 2, 16, 14, 8, 4, 12, 15, 13, 10],
               [18, 11, 9, 17, 1, 6, 14, 3, 7, 5, 8, 15, 19, 2, 4, 13, 16, 12, 10]]
        for line in arr:
            self.checkMagicHexa(line)

        return
    def test(self):
        """
        docstring for test
        """
        # self.getAllSquareFour()
        # self.testRotateMatrix()
        # self.testMagicHexa()
        

        return

magic = MagicSquare()
# magic.setOrder(7)
# magic.magicSquare1()
magic.test()