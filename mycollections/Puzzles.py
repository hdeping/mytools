#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-02-29 17:25:40
    @project      : solutions for math puzzles
    @version      : 1.0
    @source file  : Puzzles.py

============================
"""

from sympy import *
import itertools
import numpy as np


class Puzzles():
    """
    solutions for math puzzles
    """
    def __init__(self):
        super(Puzzles, self).__init__()
    def abSeven(self):
        """
        docstring for abSeven
        """
        arr = np.arange(100)
        lines = itertools.combinations(arr,2)

        num = 7**7
        for (a,b) in lines:
            p1 = (a%7 == 0)
            p2 = (b%7 == 0)
            p3 = ((a+b)%7 == 0)
            if p1 or p2 or p3:
                continue

            target = (a+b)**7 - a**7 - b**7
            if target%num == 0:
                print(a,b,target,num)

        return

    def test(self):
        """
        docstring for test
        """
        self.abSeven()
        return

puzzle = Puzzles()
puzzle.test()
        
