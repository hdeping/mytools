#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-08-20 07:43:10
    @project      : The module for drawing curves, histgrams, 
                    scatter points and so on.
    @version      : 0.1
    @source file  : DrawCurve.py

============================
"""

import numpy as np
import matplotlib 
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import sys
import os

class DrawCurve():
    """docstring for DrawCurve
    It is used for visualization of scientific data
    """
    def __init__(self):
        """
        self.lineList:
            list for the line types
        self.typeNum:
            length of self.lineList
        self.colorList:
            list for the colors
        self.colorNum:
            length of self.colorList

        the line type:
            o: circle point
            p: pegagon point
            h: hexagon point
            ^: triangal point(angle up)
            v: triangal point(angle down)
            >: triangal point(angle right)
            <: triangal point(angle left)
            -: solid line
            --: dash line
        """
        super(DrawCurve, self).__init__()
        self.lineList  = ['^-','o-','p-','h-']
        self.typeNum   = len(self.lineList)
        self.colorList = ['b','#CD6600','#FF00FF', '#00FFFF', '#808000', 
                          '#800080','r','g', '#008080','k', '#8a977b','#1d8308']
        self.colorNum  = len(self.colorList)
    def setDataFileName(self,dataFileName):
        """
        setup for the dataFileName
        """
        self.dataFileName = dataFileName
        return
    def getFileName(self):
        """
        get the filename of the output png 
        corresponding to how many images exists in the 
        current directory
        """
        fileList = os.listdir('.')
        num = 1
        for name  in fileList:
            if name.endswith(".png"):
                num += 1
        name = []
        name.append("%d_%s.png"%(num,self.dataFileName))
        name.append("%d_%s.svg"%(num,self.dataFileName))
        
        return name 
        
    def plotData(self):
        """
        load data from self.dataFileName and 
        plot the data with lines
        """
        try:
            data = np.loadtxt(self.dataFileName,delimiter=',',dtype=float)
        except UnboundLocalError:
            print("there is no file %s"%(filename))
        x = data[:,0]
        dim = len(data[0])
        for i in range(1,dim):
            y = data[:,i]
            plt.plot(x,y,self.lineList[i%self.typeNum],
                     color = self.colorList[i%self.colorNum],
                     label = 'data'+str(i),linewidth=2)
        return

    def draw(self):
        """
        function for drawing
        """
        plt.figure(1,figsize=(9,9))
        # set sizes 
        self.setSizes(32,28,24)

        # title
        self.setCurveTitle()
        # x,y labels 
        self.setCurveLabels()
        # x,y ticks
        self.setCurveTicks()
        self.plotData()
        self.setCurveLegends()
        # results output
        name = self.getFileName()
        for output in name :
            plt.savefig(output,dpi=300)
        plt.show()

    def setSizes(self,titleSize,labelSize,tickSize):
        """
        setup for the fontsizes of title, labels
        and ticks
        """
        self.titleSize  = titleSize
        self.labelSize  = labelSize
        self.tickSize   = tickSize

        return
    def setCurveTitle(self):
        """
        setup for the title of the curve
        """
        plt.title('Data',fontsize=self.titleSize)
        return

    def setCurveLabels(self):
        """
        setup for the labels of the curve
        """
        plt.xlabel('x',fontsize=self.labelSize)
        plt.ylabel('y',fontsize=self.labelSize)

        return
    def setCurveTicks(self):
        """
        setup for the ticks of the curve
        """
        plt.xticks(fontsize=self.tickSize)
        plt.yticks(fontsize=self.tickSize)
        #plt.axis([0,5,0,20]);

        return

    def setCurveLegends(self):
        """
        setup for the legends of the curve
        """
        leg = plt.legend(loc="upper center",fontsize = 20)
        ii = 0
        for text in leg.get_texts():
            ii += 1
            text.set_color(self.colorList[ii % self.colorNum])
        plt.grid(True)

        return
    def barsData(self,filename):
        """
        visual the data with bars
        """
        try:
            data = np.loadtxt(filename,delimiter=',',dtype=float)
        except UnboundLocalError:
            print("there is no file %s"%(filename))
        time    = data[:,0]
        posts   = data[:,1]
        threads = data[:,2]
        n_groups = len(time) 
        index = np.arange(n_groups)  
        opacity = 0.4  
        bar_width = 0.35  
        plt.bar(index ,posts  ,bar_width,  
                alpha=opacity,color='r',label='发帖数')

    def histData():
        """
        get a histagram
        """
        plt.figure(1,figsize=(9,9))
        plt.xlim([-5.0,5.0])
        plt.ylim([0.0,0.5])
        filename = "randn.txt"
        arr = np.loadtxt(filename,delimiter=' ',dtype=float)
        n,bins,patches = plt.hist(arr,bins=256,normed=1,edgecolor='None',facecolor='blue')
        #plt.show()
        filename = getFileName()
        for name in filename:
            plt.savefig(name,dpi=300)
        plt.show() 

    def splineData(filename):
        """
        get the B-spline of the scatter point
        """
        try:
            data = np.loadtxt(filename,delimiter=',',dtype=float)
        except UnboundLocalError:
            print("there is no file %s"%(filename))
        x = data[:,0]
        t = range(len(x))
        knots = [2, 3, 4]
        ipl_t = np.linspace(0.0, len(x) - 1, 100)
        x_tup = si.splrep(t, x, k=3, t=knots)
        x_i = si.splev(ipl_t, x_tup)

        dim = len(data[0])
        for i in range(1,dim):
            y = data[:,i]
            y_tup = si.splrep(t, y, k=3, t=knots)
            y_i = si.splev(ipl_t, y_tup)
            plt.plot(x,y,lineList[i%4],color=colorList[i%5],label=legend[i-1])
            j=i+1
            plt.plot(x_i,y_i,'-',color=colorList[j%5],label="spline")
        return 

    def test(self):
        """
        test for the DrawCurve module
        """
        if len(sys.argv) == 1:
            print("Please input a file")
        else:
            # draw 
            filename = sys.argv[1]
            self.setDataFileName(filename)
            self.draw()

# test = DrawCurve()
# test.setDataFileName("data.txt")
# test.draw()
