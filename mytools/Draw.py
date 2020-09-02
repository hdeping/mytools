#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-09-02 10:47:38
    @project      : tools from Drawing
    @version      : 1.0
    @source file  : Draw.py

============================
"""
import turtle
import os
import numpy as np
import matplotlib 
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import sys

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
class TurtlePlay():
    """
    docstring for TurtlePlay
    It is module for presentation of the usage of turtle module
    """
    def __init__(self):      
        """
        self.theta: 
            rotation angle for each time
        self.speed:
            frames per second
            such as : self.speed = 4
            the panel will be drawed 4 times in a second
        self.length:
            length of the line
        self.number:
            cycles number 

        All the above parameters were initialized to be None
        """  
        super(TurtlePlay, self).__init__()       
        self.theta = None 
        self.speed = None  
        self.length = None
        self.number = None
    
       
    def initParas(self):
        """
        initialize the parameters
        """
        
        self.setColor(['red','blue'])
        self.setParas(35,50,100,4)

        return
    
    
    
    def setColor(self,colors):
        """
        set the color 
        input: colors, array with two items
        such as : ["red","red"]
        """
        turtle.color(colors[0],colors[1])

        return 

    def setTheta(self,theta):
        """
        set the rotation angle
        """
        self.theta = theta
        return 

    
    def setLength(self,length):
        """
        set the length of the line
        """
        self.length = length
        return
    
    def setNumber(self,number):
        """
        set the number of cycles
        """
        self.number = number
        return

    def setSpeed(self,speed):
        """
        set the fps(frames per second)
        """
        self.speed = speed
        turtle.speed(speed)
        return

    def draw(self):
        """
        draw the pattern
        length is smaller and smaller in each step
        """
        i = 0
        while i < self.number:
            # print(i)
            turtle.forward(self.length - i)
            turtle.shape('turtle')
            turtle.right(self.theta)
            i += 1

    
    def writeImage(self): 
        """
        write the image into ps and pdf
        ps2pdf is a linux command which 
        converts a .ps file into a .pdf one
        """
        cv = turtle.getcanvas()
        filename = "new.ps"
        print("write to ",filename)      
        cv.postscript(file = filename,colormode='color')
        command = "ps2pdf %s"%(filename)
        print(command)
        os.system(command)

    def square(self):
        """
        draw a single square
        with the shape turtle
        """
        for i in range(4):
            turtle.forward(self.length)
            turtle.shape('turtle')
            turtle.right(self.theta)

        return
        
    
    def polygon(self,n):
        """
        draw a single polygon
        with the shape turtle
        """
        for i in range(n):
            turtle.forward(self.length)
            turtle.shape('turtle')
            turtle.right(self.theta)

        return

    def drawSquares(self):
        """
        draw squares
        add length by 3.5 after drawing a square
        """
        for i in range(self.number):
            self.square()
            turtle.right(5)
            self.length += 3.5

        return

    
    def drawPolygons(self,n):
        """
        draw polygons 
        add length by 3.5 after drawing a polygon
        """
        for i in range(self.number):
            self.polygon(n)
            turtle.right(5)
            self.length += 3.5

        return

    
    def setParas(self,theta,length,number,speed):
        """
        setup for all the parameters
        including theta, length, number and speed
        """
        self.setTheta(theta)
        self.setLength(length)
        self.setNumber(number)
        self.setSpeed(speed)

        return 
    
    
    def printParas(self):
        """
        print out all the parameters
        including theta, length, number and speed
        """
        print("theta: ",self.theta)
        print("length: ",self.length)
        print("number: ",self.number)
        print("speed: ",self.speed)

        return
    
    def test(self):
        """
        test for drawing the squares
        """
        self.setColor(['red','blue'])
        self.setParas(90, 50, 60, 8)
        # self.draw()
        self.drawSquares()
        self.writeImage()
    
    
    def testPolygon(self):
        """
        test for drawing the polygons
        n was set to be 5
        """      
        n = 5
        theta = 360/n
        self.setColor(['red','blue'])
        self.setParas(theta, 50, 60, 16)
        self.drawPolygons(n)
        self.writeImage()