#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-06 22:59:18
                    2019-10-08 00:05:50
    @project      : test turtle
    @version      : 0.1
    @source file  : main.py

============================
"""


import turtle
import os
from .TurtlePlay import TurtlePlay
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

