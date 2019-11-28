#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-11-28 18:43:39
    @project      : search for 天干地支
    @version      : 1.0
    @source file  : TianGanDiZhi.py

============================
"""
import numpy as np

class TianGanDiZhi():
    """
    天干地支
    """
    def __init__(self):
        """
        self.tiangan:
            10 elements
        self.dizhi:
            12 elements
        """
        super(TianGanDiZhi, self).__init__()
        self.tiangan = ["甲","乙","丙","丁","戊",
                        "己","庚","辛","壬","癸"]
                        
        self.tiangan = self.tiangan*6
        self.dizhi   = ["子","丑","寅","卯",
                        "辰","巳","午","未",
                        "申","酉","戌","亥"]

        self.dizhi   = self.dizhi*5
    def getYears(self):
        """
        docstring for getYears
        天干，10
        地支，12
        totally 60

        """
        self.years = []
        for line in zip(self.tiangan,self.dizhi):
            year = "".join(line)
            self.years.append(year)
        # print(self.years)
        dicts = {}
        for i,line in enumerate(self.years):
            dicts[line] = 1984 + i 
        self.dicts = dicts
        
        return
    def run(self,key):
        """
        docstring for run
        """
        self.getYears()
        if key in self.dicts:
            year  = self.dicts[key]
            years = np.arange(year,0,-60)
            print(key,years)
        else:
            print("%s is invalid"%(key))
   
        
        return
        


test = TianGanDiZhi()
key = '庚戌'
test.run(key)






