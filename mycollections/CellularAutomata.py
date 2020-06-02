#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-04-16 23:24:13
    @project      : code for cellular automata
    @version      : 1.0
    @source file  : CellularAutomata.py

============================
"""
from PIL import Image
import numpy as np
import matplotlib 
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class CellularAutomata():
    """docstring for CellularAutomata"""
    def __init__(self):
        super(CellularAutomata, self).__init__()

    def getRules(self,ruleType):
        """
        docstring for getRules
        ruleType:
            integer 
        """
        rules = []
        num = self.colors**self.positions   
        for i in range(num):
            rules.append(ruleType%self.colors)
            ruleType = ruleType//self.colors
        return rules 

    def arr2Num(self,arr):
        """
        docstring for arr2Num
        """
        res = 0 
        for i in arr:
            res = self.colors*res + i 
        return res
    def automata(self,ruleType=0,tag=None):
        """
        docstring for automata
        """
        self.colors = 3
        self.positions = 3
        num = self.colors**(self.colors**3)
        length = 1 + len(str(num))
        # rules = self.getRules(ruleType)
        # print(rules)
        rules = np.random.randint(0,self.colors,self.colors**3)
        ruleType = self.arr2Num(rules)
        n = 501
        m = n

        image = np.ones((m,n,3),np.uint8)
        types =[[0,0,0],
                [255,255,255],
                [128,128,128]]
        types =[[255,255,0],
                [0,255,255],
                [255,0,255]]
        # types =[[255,0,0],
        #         [0,0,255]]

        celluar = np.zeros(n,np.uint8)
        celluar[n//2] = 1
        # random initialized first row

        # celluar = np.random.randint(0,self.colors,n)
        # tag     = self.arr2Num(celluar)


        for i in range(n):
            image[0,i,:] = types[celluar[i]]

        coefs = np.array([self.colors**2,self.colors,1])
        for i in range(1,m):
            tmp = celluar.copy()
            # first column
            num = coefs[1]*tmp[0]+coefs[2]*tmp[1]
            num = rules[num]
            celluar[0] = num 
            image[i,0,:] = types[num]
            # middle columns 
            for j in range(1,n-1):
                num  = sum(coefs*tmp[j-1:j+2])
                num = rules[num]
                celluar[j] = num 
                image[i,j,:] = types[num]
            # last column
            num = coefs[0]*tmp[-2]+coefs[1]*tmp[-1]
            num = rules[num]
            celluar[-1] = num 
            image[i,-1,:] = types[num]

        image = Image.fromarray(image)
        name = str(ruleType)
        name = "0"*(length-len(name))+name
        if tag is None:
            imageName = "automata%s_%d.png"%(name,self.colors)
        else:
            imageName = "automata%s_%d.png"%(name,tag)

        print("save to ",imageName)
        image.save(imageName)


        return

    def testAutomata(self):
        """
        docstring for testAutomata
        """
        begin = 50
        end   = 256
        # for i in range(begin,end):
        sets = [57,73,86,150,165,
                126,52,129,99,109]

        np.random.seed(2)
        # for i in sets:
        #     self.automata(ruleType=i)

        for i in range(300):
            self.automata(ruleType=129)


        return
    def test(self):
        """
        docstring for test
        """
        self.testAutomata()
        return

celluar = CellularAutomata()
celluar.test()
        