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
    def __init__(self,parameters):
        super(Mandelbrot, self).__init__()
        # usually a complex number
        self.C          = parameters["C"]
        # maximum iteration number
        self.maxIter    = parameters["maxIter"]
        # function range
        self.func_range = parameters["func_range"]
        # threshold
        self.threshold  = parameters["threshold"]
        
        # split number
        self.n          = parameters["n"]
        self.filename   = parameters["filename"]

        self.Z = 0.0
        
    def get_iteration(self):
        # print(i)
        iteration = 0 
        while abs(self.Z)**2 < self.threshold and \
              iteration < self.maxIter:
            self.Z = self.Z**2 + self.C
            iteration += 1
        return iteration
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
        data = (xValue,yValue,iterations,self.maxIter)
        print("get plotting")
        
        plot(self.filename,data)

parameters = {}
parameters["C"] = 2.0
parameters["maxIter"] = 80
parameters["threshold"] = 2.0
parameters["func_range"] = [-5,5,-5,5]
parameters["n"] = 256
parameters["filename"] = "mandelbrot.txt"

mandelbrot = Mandelbrot(parameters)

