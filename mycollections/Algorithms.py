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

class QTable():
    """
    there are 16 states
    0  1  2  3
    4  5  6  7
    8  9  10 11
    12 13 14 15

    reward = 0
    reward6 = reward9 = - 1
    reward10 = 1

    """
    def __init__(self):
        """
        self.qTable:
            table for the q values
        self.reward:
            reward of each site in the game
        self.epsilon:
            (0,1) probability, best or random
        self.gamma:
            decay factor (0,1)
        self.lr:
            learning rate
        value = r_{t+1}+gamma*max q(s_{t+1},a') - q(s,a)
        q(s,a) = q(s,a) + lr*value
        """
        super(QTable, self).__init__()
        #  get random seed
        np.random.seed(int(time.time()))

        # initialize the value array
        self.reward = np.zeros(16)
        self.reward[6] = -1
        self.reward[9] = -1
        self.reward[10] = 1
        self.reward[14] = -1
        self.actions = ['left', 'right', 'down', 'up']
        # hyper parameters
        self.epsilon = 0.9
        self.gamma   = 0.9
        self.lr      = 0.1
        self.qTable  = None

        return
    def initQValue(self):
        """
        not all the positive share the same kinds of operations
        some positions only have two ones, some others have three

        total operations: 4*16 - 4*4 = 48
        48 action-value   Q values
        
        4*2 + 8*3 + 4*4 = 48
        QTable = np.zeros((16,4))
        updating states
        0,1 left,right -/+ 1
        2,4 up,down    -/+ 4

        """
        # (ii,jj) 4*ii + jj
        table = []
        # four actions of all states
        for i in range(16):
            dictionary = {}
            for j in range(4):
                dictionary[self.actions[j]] = 0
            table.append(dictionary)

        for i in range(4):
            # fist row
            state = i
            table[state].pop('up')
            # fourth row
            state = 12 + i
            table[state].pop('down')
            # fist column
            state = i * 4
            table[state].pop('left')
            # fourth column
            state = i * 4 + 3
            table[state].pop('right')

        return table

    def tablePrint(self):
        """
        print the self.qTable
        """
        string = ""
        for i, line in enumerate(self.qTable):
            #print(i, line.values())

            string = "%d,"%(i)
            values = [value for value in line.values()]
            for value in line.values():
                string = "%s,%.3f,"%(string,value)
            print(string)
        return
    def updateStates(self,input_state, action):
        """
        0  1  2  3  no up
        0  4  8  12 no left
        3  7  11 15 no right
        12 13 14 15 no down
        
        """
        if action == 'left':
            state1 = input_state - 1
        elif action == 'right':
            state1 = input_state + 1
        elif action == 'down':
            state1 = input_state + 4
        elif action == 'up':
            state1 = input_state - 4

        assert state1 >= 0 and state1 <= 15

        #print("updateStates",action,state1,"input state",input_state)
        return state1

    def updateValues(self,input_state):
        """
        update the Q value with the Bellman equation
        value = r_{t+1}+gamma*max q(s_{t+1},a') - q(s,a)
        q(s,a) = q(s,a) + lr*value
        """
        q_table = self.qTable[input_state]
        for action in q_table:
            #print("action",action,input_state)
            new_state = self.updateStates(input_state, action)
            line = self.qTable[new_state]
            # action list
            items = line.items()
            # value list
            actions = [key for key, value in items]
            values = [value for key, value in items]
            #values = np.array(values)

            # Bellmann equation
            value = self.reward[new_state] + self.gamma * max(values)
            q_table[action] += self.lr * (value - q_table[action])


    def getNewState(self,input_state):
        #
        #  update the values
        self.updateValues(input_state)
        line = self.qTable[input_state]
        items = line.items()
        # value list
        actions = [key for key, value in items]
        values =  [value for key, value in items]
        values = np.array(values)

        if np.random.rand() < self.epsilon:
            # get the best action
            action_index = np.argmax(values)
        else:
            # get the random action
            action_index = np.random.randint(len(line))

        # get the action

        action = actions[action_index]
        state = self.updateStates(input_state, action)
        return state

    def run(self):
        """
        docstring for run

        """
        self.qTable = self.initQValue()
        
        result = []
        cycles = 10000
        for epoch in range(cycles):
            #print("epoch", epoch)
            state = 0
            while state not in [6,9,10,14]:
                state = self.getNewState(state)
                #print("state", state)
            if epoch > cycles*0.8:
                self.epsilon = 1.0
            #print(state)
            result.append(state)
        self.writeResult(result)

        return
    def writeResult(self,result):
        """
        docstring for writeResult
        write the final into a file
        """
        result = np.array(result)
        result = np.reshape(result,(100,100))

        filename = "data.txt"
        output = np.zeros((100,2))
        for i,arr in enumerate(result):
            ii = sum(arr==10)
            print(i,ii)
            output[i,:] = [i,ii]
            
        np.savetxt(filename,output,fmt="%d,%d")
        self.tablePrint()
        
        return
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
        # arr1 = np.array(arr1)
        # res = np.zeros(num1+num2-1,int)
        res = []
        for i in range(num1+num2-1):
            res.append(0)

        for index,i in enumerate(arr2):
            # res[index:index+num1] += arr1*i
            for j in range(num1):
                res[j+index] += arr1[j]*i

        return res

    def polynomialDivide(self,arr1,arr2):
        """
        docstring for polynomialDivide
        arr1[0] = 1 and arr2[0] = 1
        both arr1 and arr2 are infinely Tayler series
        return:
            arr1/arr2, division of the series

        """
        assert(arr1[0] == 1)
        assert(arr2[0] == 1)

        res = [1]

        n = len(arr1)-1
        for i in range(1,n):
            num = arr1[i]
            for j in range(i):
                num = num - arr2[j+1]*res[i-j-1]
            res.append(num)
            
        return res
    def polynomialFactor(self,arr1,arr2):
        """
        docstring for polynomialFactor
        (a1x^n+...) = (b1x+c1)*(k1x^(n-1)) + K
        """
        assert(len(arr2) == 2)

        res = [arr1[0]/arr2[0]]
        for i in range(1,len(arr1)-1):
            num = (arr1[i] - arr2[1]*res[-1])/arr2[0]
            res.append(num)
        remain = arr1[-1] - res[-1]*arr2[1]
        return res,remain
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
        
