#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-09 22:45:14
    @project      : get the lines of python code 
    @version      : 0.1
    @source file  : getLines.py

============================
"""
import numpy as np
import os


class GetLines():
    """docstring for GetLines"""
    def __init__(self):
        super(GetLines, self).__init__()

    def splitNames(self):
        """
        docstring for splitNames
        split filenames into 2d array
        """
        
        results = []
        index = 0
        num = 1000
        while index < len(self.filenames):
            results.append(self.filenames[index:index+num])
            index += num
        self.filenames = results
        return
    def run(self):
        """
        docstring for run
        get the total lines of the code
        """
        self.filenames = np.loadtxt("mv",str)
        self.splitNames()
        print(len(self.filenames))
        # print(self.filenames)

        results = []
        for line in self.filenames:
            command = "wc -l %s"%(" ".join(line))
            value   = self.getValue(command)
            results.append(value)

        print(sum(results))
        
            
        return
    def getValue(self,command):
        """
        docstring for getValue
        input: command, a linux command based on wc
        return: total lines of the code
        """
        
        output = os.popen(command).read()
        output = output.split("\n")
        output = output[-2]
        output = output.split(" ")
        output = int(output[-2])
        print(output)
        return output
