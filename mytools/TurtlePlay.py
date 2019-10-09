#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-07 00:10:36
                    2019-10-09 10:15:11
    @project      : turtle animation
    @version      : 1.0
    @source file  : TurtlePlay.py

============================
"""


import turtle
import os


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
