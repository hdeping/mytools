#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-07 00:10:36
    @project      : turtle animation
    @version      : 1.0
    @source file  : main.py

============================
"""


import turtle
import os


class TurtlePlay():
    """docstring for TurtlePlay"""
    def __init__(self):
        super(TurtlePlay, self).__init__() 
        # theta: rotation angle for each time
        self.theta = None 
        # frames per second
        # such as : self.speed = 4
        # the panel will be drawed 4 times in a second
        self.speed = None
        # length of the line
        self.length = None
        # cycles number 
        self.number = None
    
    # initialize the parameters   
    def initParas(self):
        self.setColor(['red','blue'])
        self.setParas(35,50,100,4)

        return
    # set the color for 
    # input: colors, array with two items
    # such as : ["red","red"]
    def setColor(self,colors):
        turtle.color(colors[0],colors[1])

        return 
    # set the rotation angle
    def setTheta(self,theta):
        self.theta = theta
        return 

    # set the length of the line
    def setLength(self,length):
        self.length = length
        return
    # set the number of cycles
    def setNumber(self,number):
        self.number = number
        return

    # set the fps(frames per second)
    def setSpeed(self,speed):
        self.speed = speed
        turtle.speed(speed)
        return

    # draw the pattern
    def draw(self):
        i = 0
        while i < self.number:
            # print(i)
            turtle.forward(self.length - i)
            turtle.shape('turtle')
            turtle.right(self.theta)
            i += 1

    # write the image into ps and pdf
    # ps2pdf is a linux command which 
    # converts a .ps file into a .pdf one
    def writeImage(self):    

        cv = turtle.getcanvas()
        filename = "new.ps"
        print("write to ",filename)
        
        cv.postscript(file = filename,colormode='color')
        
        command = "ps2pdf %s"%(filename)
        print(command)
        os.system(command)

    # draw a square
    def square(self):
        for i in range(4):
            turtle.forward(self.length)
            turtle.shape('turtle')
            turtle.right(self.theta)

        return
        
    # draw a polygon
    def polygon(self,n):
        for i in range(n):
            turtle.forward(self.length)
            turtle.shape('turtle')
            turtle.right(self.theta)

        return

    # draw squares
    def drawSquares(self):
        for i in range(self.number):
            self.square()
            turtle.right(5)
            self.length += 3.5

        return

    # draw polygons 
    def drawPolygons(self,n):
        for i in range(self.number):
            self.polygon(n)
            turtle.right(5)
            self.length += 3.5

        return

    # set all the parameters
    def setParas(self,theta,length,number,speed):
        self.setTheta(theta)
        self.setLength(length)
        self.setNumber(number)
        self.setSpeed(speed)

        return 
    
    # print out all the parameters
    def printParas(self):
        print("theta: ",self.theta)
        print("length: ",self.length)
        print("number: ",self.number)
        print("speed: ",self.speed)

        return
    
    # test for drawing the squares  
    def test(self):
        self.setColor(['red','blue'])
        self.setParas(90, 50, 60, 8)
        # self.draw()
        self.drawSquares()
        self.writeImage()
    
    # test for drawing the polygons
    def testPolygon(self):
        n = 5
        theta = 360/n
        self.setColor(['red','blue'])
        self.setParas(theta, 50, 60, 16)
        self.drawPolygons(n)
        self.writeImage()
