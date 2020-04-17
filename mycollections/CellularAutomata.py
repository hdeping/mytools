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

    def automata(self,ruleType=0):
        """
        docstring for automata
        """
        self.colors = 2
        self.positions = 3
        rules = self.getRules(ruleType)
        print(rules)
        n = 1025
        n = 501
        m = n

        image = np.ones((m,n,3),np.uint8)
        types =[[0,0,0],
                [255,255,255],
                [128,128,128]]
        # types =[[255,0,0],
        #         [0,0,255]]

        celluar = np.zeros(n,np.uint8)
        celluar[n//2] = 1 
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
        name = "0"*(4-len(name))+name
        imageName = "automata%s.png"%(name)
        print("save to ",imageName)
        image.save(imageName)


        return

    def testAutomata(self):
        """
        docstring for testAutomata
        """
        begin = 50
        end   = 256
        for i in range(begin,end):
            self.automata(ruleType=i)
        return
    def test(self):
        """
        docstring for test
        """
        self.testAutomata()
        return

celluar = CellularAutomata()
celluar.test()
        