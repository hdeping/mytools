#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-05-29 20:24:08
    @project      : mandelbrot
    @version      : 0.1
    @source file  : main.py

============================
"""

from plot import plot
import numpy as np
from tqdm import tqdm



class Mandelbrot():
    """docstring for Mandelbrot"""
    def __init__(self,C,maxIter,
                      func_range,threshold,
                      n,filename):
        super(Mandelbrot, self).__init__()
        # usually a complex number
        self.C = C
        # maximum iteration number
        self.maxIter = maxIter
        # function range
        self.func_range = func_range
        # threshold
        self.threshold = threshold
        self.Z = 0.0
        # split number
        self.n = n
        self.filename = filename
        
    def get_iteration(self):
        # print(i)
        iteration = 0
        cValue = 0
        # while abs(self.Z)**2 < self.threshold and \
        #       iteration < self.maxIter:
        #     # self.Z = self.Z**2 + self.C
        #     self.Z = self.Z**2 +  self.Z
        #     iteration += 1

        while abs(cValue)**2 < self.threshold and \
              iteration < self.maxIter:
            # cValue = cValue**2 - cValue +  self.Z
            cValue = cValue**2 +  self.Z
            iteration += 1

        # if iteration > self.maxIter//2 - 1:
        #     return 1
        # else:
        #     return 0
        # iteration /= self.maxIter
        # if iteration > 0.5:
        #     return 0
        # else:
        #     return (iteration - 0.5)*2
        return iteration/self.maxIter
    def get_mandelbrot(self):
        xValue = np.linspace(self.func_range[0],
                             self.func_range[1],
                             self.n)
        yValue = np.linspace(self.func_range[2],
                             self.func_range[3],
                             self.n)
        iterations = np.zeros((self.n,self.n))
        # get iterations
        print("calculating data")
        
        for i,x in tqdm(enumerate(xValue)):
            for j,y in enumerate(yValue):
                self.Z = x + y*1.0j
                iterations[i,j] = self.get_iteration()

        # get plot
        xValue = np.arange(self.n)
        yValue = np.arange(self.n)
        data = (xValue,yValue,iterations.transpose())
        # data = (xValue,yValue,iterations,self.maxIter)
        print("get plotting")
        
        plot(self.filename,data)

