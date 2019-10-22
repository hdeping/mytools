#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-05-29 20:24:08
                    2019-10-22 20:13:39
    @project      : mandelbrot
    @version      : 1.0
    @source file  : Mandelbrot.py

============================
"""

from plot import plot
import numpy as np
from tqdm import tqdm


class Mandelbrot():
    """
    docstring for Mandelbrot
    Z_n = Z_n^2 + C
    |Z_n| < \delta

    usage:
        mandel = Mandelbrot()
        mandel.run()
    """
    def __init__(self,parameters):
        super(Mandelbrot, self).__init__()
        self.setParas(parameters)
        
    def setParas(self,parameters):
        """
        docstring for setParas
        self.C:
            usually a complex number
        self.maxIter:
            maximum iteration number
        self.func_range:
            function range
        self.threshold:
            threshold of the system
        self.n:
            split number
        self.filename:
            output file name 
        self.Z:
            initial value, usually 0.0
        """
        
        self.C          = parameters["C"]  
        self.maxIter    = parameters["maxIter"]
        self.func_range = parameters["func_range"]
        self.threshold  = parameters["threshold"]
        self.n          = parameters["n"]
        self.filename   = parameters["filename"]
        self.Z = 0.0
        return   
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
        
        # plot(self.filename,data)
    def run(self):
        """
        docstring for run
        run an instance
        """
        parameters = {}
        parameters["C"] = 2.0
        parameters["maxIter"] = 80
        parameters["threshold"] = 2.0
        parameters["func_range"] = [-5,5,-5,5]
        parameters["n"] = 256
        parameters["filename"] = "mandelbrot.txt"
        self.setParas(parameters)
        self.get_mandelbrot()
        return



