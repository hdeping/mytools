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

    def abcDelta(self):
        """
        docstring for abcDelta
        a,b,c are integers,
        s2 = sqrt(2)
        s3 = sqrt(3)
        abs(a+b*s2+c*s3) < 10^{-11}

        b^2 - 6c^2 = 3, b should be even
        b,c = (3,1)
        """

        solutions = [[1,0],[5,2]]

        for i in range(10):
            arr = []
            for j in range(2):
                num = solutions[-1][j]*10 - solutions[-2][j]
                arr.append(num)
            print(i+3,arr,arr[0]**2 - 6*arr[1]**2)
            solutions.append(arr)
        for i in range(10):
            a = (solutions[i][0] + solutions[i+1][0])//2
            b = (solutions[i][1] + solutions[i+1][1])//2
            print(a,b,a**2-6*b**2)
        for i in range(1000):
            b = 6*i**2 -1 
            b = sqrt(Integer(b))
            if b.is_integer:
                print(b,i)
        
        return
    def test(self):
        """
        docstring for test
        """
        # self.abSeven()
        self.abcDelta()
        return

puzzle = Puzzles()
puzzle.test()
        
