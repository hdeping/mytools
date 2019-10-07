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
        self.theta = None 
        self.speed = None
        self.length = None
        self.number = None
        
    def initParas(self):
        self.setColor(['red','blue'])
        self.setParas(35,50,100,4)

        return
    def setColor(self,colors):
        turtle.color(colors[0],colors[1])

        return 
    def setTheta(self,theta):
        self.theta = theta
        return 

    def setLength(self,length):
        self.length = length
        return
    def setNumber(self,number):
        self.number = number
        return

    def setSpeed(self,speed):
        self.speed = speed
        turtle.speed(speed)
        return

    def draw(self):
        i = 0
        while i < self.number:
            # print(i)
            turtle.forward(self.length - i)
            turtle.shape('turtle')
            turtle.right(self.theta)
            i += 1


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

    def drawSquare(self):
        for i in range(self.number):
            self.square()
            turtle.right(5)
            self.length += 3.5

        return

    def drawPolygon(self,n):
        for i in range(self.number):
            self.polygon(n)
            turtle.right(5)
            self.length += 3.5

        return

    def setParas(self,theta,length,number,speed):
        self.setTheta(theta)
        self.setLength(length)
        self.setNumber(number)
        self.setSpeed(speed)

        return 

    def printParas(self):
        print("theta: ",self.theta)
        print("length: ",self.length)
        print("number: ",self.number)
        print("speed: ",self.speed)

        return
        
    def test(self):
        self.setColor(['red','blue'])
        self.setParas(90, 50, 60, 8)
        # self.draw()
        self.drawSquare()
        self.writeImage()
    
    def testPolygon(self):
        n = 5
        theta = 360/n
        self.setColor(['red','blue'])
        self.setParas(theta, 50, 60, 16)
        self.drawPolygon(n)
        self.writeImage()
