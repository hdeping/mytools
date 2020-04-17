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

    def automata(self,ruleType=0):
        """
        docstring for automata
        """
        ruleType = 33
        n = 1025
        ruleType = bin(ruleType)
        image = np.ones((n,n,3),np.uint8)
        types =[[0,0,0],
                [255,255,255]]

        celluar = np.zeros((n,n),np.uint8)
        celluar[0,n//2] = 1 
        return
    def test(self):
        """
        docstring for test
        """
        self.automata()
        return

celluar = CellularAutomata()
celluar.test()
        