#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-01-16 08:28:00
    @project      : compute the volume of urine
    @version      : 0.1
    @source file  : new.py

============================
"""
import numpy as np

class Urine():
    """
    docstring for Volume
    lengths were measured by a ruler
    volumes were computed by the formula for 
    the volume of a circular truncated cone
    """
    def __init__(self):
        super(Urine, self).__init__()
        
    def getVolume(self,r,R,l):
        """
        docstring for getVolume
        """
        h = (R-r)**2
        h = np.sqrt(l*l - h)
        volume = np.pi*(R*R + R*r+r*r)*h/3
        return volume

    def getVolume2(self,l1):
        """
        docstring for getVolume
        R1 = (l1*R + l2*r)/(l1+l2)
        """
        r  = 2.5
        l  = 15.6
        R  = 4.3
        l2 = l - l1 
        R1 = (l1*R + l2*r)/l
        volume = self.getVolume(r,R1,l1)
        return volume

    def getRadius(self,a,b,c):
        """
        docstring for getRadius
        """
        p = (a+b+c)/2 
        s = p*(p-a)*(p-b)*(p-c)
        s = np.sqrt(s)
        R = a*b*c/(4*s)
        return R

    def test(self):
        """
        docstring for test
        """

        mine = np.array([144.0,144.0,144.2,144.2,144.0])
        total = np.array([146.1,146.1,146.1,146.3,146.1])

        volume = np.average(total) - np.average(mine)
        volume = volume/2 * 1000
        print("体重法，体积1: %.1f mL"%(volume))


        sizes = [3.8,14.7,14.7]
        sizes = np.array(sizes)
        print(np.prod(sizes))

        print(self.getVolume(1,1,1))
        # unit: cm
        r = 2.5

        # 13.5,15.6

        R = self.getRadius(8,4.5,5.1)
        print(R)
        l = 15.6

        v1 = self.getVolume2(13.5)
        v2 = self.getVolume2(10.4)
        print("量杯法，总体积: %.1f mL"%(v1+v2))

        return

volume = Urine()
volume.test()
