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

        return
    
       
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
        return

    
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

        return

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

        return
    
    
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

        return

# draw a pig
class DrawPig(TurtlePlay):
    """docstring for DrawPig"""
    def __init__(self):
        super(DrawPig, self).__init__()
    def drawPigInit(self):
        turtle.pensize(4)
        turtle.hideturtle()
        turtle.colormode(255)
        turtle.color(self.bodyColor, self.faceColor)
        turtle.setup(840, 500)
        turtle.speed(40)

        return

    def drawPigNoses(self):
        turtle.pu()
        turtle.goto(-100,100)
        turtle.pd()
        turtle.seth(-30)
        turtle.begin_fill()
        a = 0.4
        for i in range(120):
            if 0 <= i < 30 or 60 <= i < 90:
                a = a+0.08
                turtle.lt(3)  # 向左转3度
                turtle.fd(a)  # 向前走a的步长
            else:
                a = a-0.08
                turtle.lt(3)
                turtle.fd(a)
                turtle.end_fill()

        lines = [[90,25],[0,10]]
        self.drawPigNose(lines)

        lines = [[0,20]]
        self.drawPigNose(lines)        

        return
    def drawPigNose(self,lines):
        self.drawLines(lines)
        turtle.pencolor(self.bodyColor)
        turtle.seth(10)
        self.fillRegion(5,color = self.noseColor)

        return
    def fillRegion(self,radius,color=None):
        turtle.begin_fill()
        turtle.circle(radius)
        if color:
            turtle.color(color)
        turtle.end_fill()

        return

    def drawPigHead(self):
        turtle.color(self.bodyColor, self.faceColor)
        lines = [[90,41],[0,0]]
        self.drawLines(lines)
        turtle.begin_fill()
        turtle.seth(180)
        centers = [[300, -30],[100, -60],[80, -100],
                   [150, -20],[60, -95]]
        self.drawCircles(centers)
        turtle.seth(161)
        turtle.circle(-300, 15)
        turtle.pu()
        turtle.goto(-100, 100)
        turtle.pd()
        turtle.seth(-30)
        a = 0.4
        for i in range(60):
            if 0 <= i < 30 or 60 <= i <90:
                a = a+0.08
                turtle.lt(3)  # 向左转3度
                turtle.fd(a)  # 向前走a的步长
            else:
                a = a-0.08
                turtle.lt(3)
                turtle.fd(a)
                turtle.end_fill()
        return

    def drawPigEars(self):
        turtle.color(self.bodyColor, self.faceColor)
        lines = [[90,-7],[0,70]]
        centers = [[-50, 50],[-10, 120],[-50, 54]]
        self.drawPigEar(lines, centers)
        

        lines = [[90,-12],[0,30]]
        centers = [[-50, 50],[-10, 120],[-50, 56]]
        self.drawPigEar(lines, centers)
        return

    def drawPigEar(self,lines,centers):
        self.drawLines(lines)
        self.fillEar(centers)
        
        return
    def fillEar(self,centers):
        turtle.begin_fill()
        turtle.seth(100)
        self.drawCircles(centers)
        turtle.end_fill()

        return
    def drawPigMouth(self):
        turtle.color(self.mouthColor)     
        lines = [[90,15],[0,-100]]
        self.drawLines(lines)
        turtle.seth(-80)
        centers = [[30, 40],[40, 80]]
        self.drawCircles(centers)

        return

    def drawPigEyes(self):
        lines = [[90,-20],[0,-95]]
        self.drawPigEye(lines)

        lines = [[90,-25],[0,40]]
        self.drawPigEye(lines)

        return
    def drawPigEye(self,lines):
        turtle.color(self.bodyColor, "white")
        self.drawLines(lines)
        self.fillRegion(15)

        turtle.color("black")
        lines = [[90,12],[0,-3]]
        self.drawLines(lines)
        self.fillRegion(3)
        return

    def setPigColors(self):
        self.bodyColor = (255, 155, 192)
        self.feetColor = (240,128,128)
        self.faceColor = "pink"
        self.noseColor = (160, 82, 45)
        self.mouthColor = (239, 69, 19)

        return
   
    def drawPigCheek(self):
        turtle.color(self.bodyColor)
        lines = [[90,-95],[0,65]]
        self.drawLines(lines)
        self.fillRegion(30)

        return
    def drawPigBody(self):
        turtle.color("red", (255, 99, 71))
        lines = [[90,-20],[0,-78]]
        self.drawLines(lines)

        heights = [-130,90,-135]
        centers = [[[100,10],[300,30]],
                   [[300,30],[100,3]],
                   [[-80,63],[-150,24]]]
        turtle.begin_fill()
        i = 0
        self.drawBody(heights[i], centers[i])
        self.drawLines([[0,230]])
        i = 1
        self.drawBody(heights[i], centers[i])
        turtle.color(self.bodyColor,(255,100,100))
        i = 2
        self.drawBody(heights[i], centers[i])
        turtle.end_fill()

        return

    def drawBody(self,height,centers):
        turtle.seth(height)
        self.drawCircles(centers)

        return

    def drawPigHands(self):
        turtle.color(self.bodyColor)
        lines = [[[90,-40],[0,-27]],
                 [[90,15],[0,0]],
                 [[90,30],[0,237]],
                 [[90,20],[0,0]]
        ]
        centers = [[[300,15]],
                   [[-20,90]],
                   [[-300,15]],
                   [[20,90]]
        ]
        heights = [-160,-10,-20,-170]
        for i in range(4):
            self.drawLines(lines[i])
            self.drawBody(heights[i],centers[i])

        return
    def drawPigFeet(self):
        lines = [[[90,-75],[0,-180]],
                 [[90,40],[0,90]]
        ]

        for i in range(2):
            self.drawPigFoot(lines[i])

        return

    def drawPigFoot(self,lines):
        turtle.pensize(10)
        turtle.color(self.feetColor)
        self.drawLines(lines)
        turtle.seth(-90)
        turtle.fd(40)
        turtle.seth(-180)
        turtle.color("black")
        turtle.pensize(15)
        turtle.fd(20)

        return
    def drawPigTail(self):
        turtle.pensize(4)
        turtle.color(self.bodyColor)
        lines = [[90,70],[0,95]]
        self.drawLines(lines)
        turtle.seth(0)
        centers = [[70, 20],[10, 330],[70, 30]]
        self.drawCircles(centers)

        return 

    def drawLines(self,lines):
        """
        input: lines, 2d array
        such as : [[0,90],[90,2]...]
        """
        turtle.pu()
        for line in lines:
            turtle.seth(line[0])
            turtle.fd(line[1])
        turtle.pd()
        return 
    def drawCircles(self,centers):
        for center in centers:
            turtle.circle(center[0],center[1])
        return
    def cutePig(self):
        # draw the noses, head, ears
        # eyes, cheek, mouth, body, hands
        # feet and tail of the pig
        self.setPigColors()
        self.drawPigInit()
        self.drawPigNoses()
        self.drawPigHead()
        self.drawPigEars()
        self.drawPigEyes()
        self.drawPigCheek()
        self.drawPigMouth()
        self.drawPigBody()
        self.drawPigHands()
        self.drawPigFeet()
        self.drawPigTail()
        self.writeImage()
        turtle.done()

        return