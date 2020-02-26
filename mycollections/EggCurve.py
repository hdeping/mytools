#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-11-13 20:25:57
    @project      : egg curve
    @version      : 0.1
    @source file  : eggCurve.py

============================
"""

import sympy
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np



class EggCurve():
    """
    get the analytic and numeric solutioin of 
    egg curve
    """
    def __init__(self):
        """
        self.a:
            the sum is 2*a
        self.c:
            two points (-c,0) and (c,0)
        self.beta:
            l2+beta*l1 = 2*a (egg curve, a ellepic one 
            when beta = 1)
        """
        super(EggCurve, self).__init__()
        self.a    = 2 
        self.c    = 1
        self.beta = 100
        self.count = 0
        return 

    def setParas(self,a,c,beta):
        """
        docstring for setParas
        """
        self.a    = a   
        self.c    = c   
        self.beta = beta

        return
    def analyticSol(self):
        """
        docstring for analyticSol
        """
        
        # formula:
        #    (1-beta**2)*l1**2+(4*a*beta - 4*c*cos(theta))*l1+
        #    4*c**2 - 4*a**2 = 0

        a    = sympy.Symbol("a")
        c    = sympy.Symbol("c")
        x    = sympy.Symbol("x")
        beta = sympy.Symbol("beta")
        theta = sympy.Symbol("theta")
        print(a)
        print(beta)
        equation  = (1-beta**2)*x**2
        equation += (4*a*beta - 4*c*sympy.cos(theta))*x
        equation += 4*c**2 - 4*a**2
        y = sympy.solve(equation,x)
        y = sympy.simplify(y)
        print(sympy.latex(y[0]))
    def numSol(self):
        """
        docstring for numSol
        """
        num = 500
        x = np.zeros(num*2)
        x[:num] = np.linspace(-1,0.9,500)
        x[num:] = np.linspace(0.9,1,500)
        y = []
        for t in x:
            y.append(self.getEggValue(t))
        y = np.array(y)
        plt.axis("equal")
        plt.plot(y[:,0],y[:,1])
        plt.plot(y[:,0],-y[:,1])
        return
    def showImage(self):
        """
        docstring for showImage
        """
        # plt.grid(True)
        name = "new%d.png"%(self.count)
        print(name)
        plt.savefig(name,dpi=400)
        plt.show()
        return
    def getEggValue(self,t):
        """
        docstring for getEggValue
        t = cos(theta)
        return:
            (x,y) <-- (l1*cos(theta) - c, l1*sin(theta))
        """
        # get l1
        A = self.beta**2 - 1 
        B = self.a*self.beta - self.c*t 
        C = self.a*self.a - self.c*self.c 
        delta = np.sqrt(B*B - A*C)
        L = []
        L.append(2*(B + delta)/A)
        L.append(2*(B - delta)/A)
        # if L[0] > 0 and L[0] < self.a:
        #     l1 = L[0]
        # else:
        #     l1 = L[1]
        l1 = min(L)
        if l1 < 0:
            l1 = sum(L) - l1
            

        # if t > 0.9:
        #     print(L)
        x = l1*t - self.c
        y = l1*np.sqrt(1-t*t)
        
        return [x,y]
    def run(self):
        """
        docstring for run
        """
        # for i in range(50):
        #     self.beta  = 0 + 0.02*i
        #     self.count = i+1
        #     self.numSol()
        for i in range(100):
            self.beta  = 1.0 + 0.1*(i+1)
            self.count = i+1
            self.numSol()
        self.showImage()
        return
