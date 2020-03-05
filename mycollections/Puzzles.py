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


    def nSquare(self):
        """
        docstring for nSquare
        """
        solutions = [[1,0],[2,1]]
        lambda1 = 2*solutions[1][0]
        for i in range(10):
            arr = []
            for j in range(2):
                num = solutions[-1][j]*lambda1 - solutions[-2][j]
                arr.append(num)
            # print(i+2,arr,arr[0]**2 - 7*arr[1]**2)
            solutions.append(arr)
            if i%2 == 0:
                n = 2*arr[0]+2
                print(i,n,sqrt(n))
            
        return

    def fixedPointConic(self):
        """
        docstring for fixedPointConic
        """
        a,b,k,m,l = symbols("a b k m,l")
        x = symbols("x0:3")
        y = symbols("y0:3")

        s = (y[2] - y[0])*(y[1] - y[0])
        s = s - l*((x[2] - x[0])*(x[1] - x[0]))
        print(expand(s))

        # x1*x2, x1 + x2 
        # l = -1
        x12  = ((a*m)**2 - (a*b)**2)/((a*k)**2+b*b)
        x1x2 = -2*a*a*k*m/((a*k)**2+b*b)
        y12  = k*k*x12 + k*m*x1x2 + m*m 
        y1y2 = k*x1x2 + 2*m 
        s = y12 - y[0]*y1y2 + y[0]**2 
        s = s - l*(x12 - x[0]*x1x2 + x[0]**2 )
        # s = s.subs(m,y[0] - k*x[0])
        s = self.simEqn(s,m)

        # print(latex(s))
        
        # xy12 = x1*y2 + x2*y1
        xy12 = 2*k*x12 + m*x1x2

        s = xy12 + 2*x[0]*y[0] - y[0]*x1x2 
        s = s - x[0]*y1y2 
        s = s - l*(x12 - x[0]*x1x2 + x[0]**2)
        # s = s.subs(m,y[0] - k*x[0])
        s = self.simEqn(s,m)
        # s = s.factor()
        s = -s
        print(latex(s))
        print(s)
        x0,y0 = x[0],y[0]
        m1 = -2*a**2*k*l*x0 + 2*a**2*k*y0 - 2*b**2*x0
        m1 = m1/(a**2*l) - (y[0] - k*x[0])
        m1 = m1.expand().simplify()
        # print(latex(m1))

        m1 = m1*(y[0] - k*x[0])*a*a*l
        m1 = m1.expand()
        # print(latex(m1))   

        alpha,beta = symbols("alpha beta")
        s = (xy12 + 2*x[0]*y[0])*alpha
        s = s - alpha*(x[0]*y1y2 + y[0]*x1x2)
        s = s + beta*(y12 - y[0]*y1y2 + y[0]**2)
        s = s - l*(x12 - x[0]*x1x2 + x[0]**2 )  
        # s = s.subs(m,y[0] - k*x[0])
        s = self.simEqn(s,m) 
        print(latex(s)) 

        print(s) 

        m1 = 2*a**2*alpha*k*y0 - 2*a**2*k*l*x0
        m1 = m1 - 2*alpha*b**2*x0 - 2*b**2*beta*y0
        m1 = -m1/(-a**2*l + b**2*beta) - (y[0] - k*x[0])
        m1 = m1.expand().simplify()
        m1 = m1.collect(k)
        print(latex(m1))


        return

    def simEqn(self,s,m):
        """
        docstring for simEqn
        simplify the equation
        """
        s = s.expand().simplify()
        s,denom = fraction(s)
        s = s.expand().collect(m)
        return s


    def fixedPointParabola(self):
        """
        docstring for fixedPointParabola
        """
        p,k,m,l  = symbols("p k m,l")
        alpha,beta = symbols("alpha beta")
        x = symbols("x0:3")
        y = symbols("y0:3")

        x12  = -(2*k*m - 2*p)/(k*k)
        x1x2 = (m/k)**2
        y12  = k*k*x12 + k*m*x1x2 + m*m 
        y1y2 = k*x1x2 + 2*m 
        xy12 = 2*k*x12 + m*x1x2

        s = (xy12 + 2*x[0]*y[0])*alpha
        s = s - alpha*(x[0]*y1y2 + y[0]*x1x2)
        s = s + beta*(y12 - y[0]*y1y2 + y[0]**2)
        s = s - l*(x12 - x[0]*x1x2 + x[0]**2 )  
        s = s.subs(m,y[0] - k*x[0])
        s = self.simEqn(s,m) 
        print(latex(s)) 

        s = s.factor()
        print(latex(s))

        return
    def test(self):
        """
        docstring for test
        """
        # self.abSeven()
        # self.abcDelta()
        # self.nSquare()
        # self.fixedPointConic()
        self.fixedPointParabola()

        return

puzzle = Puzzles()
puzzle.test()
        
