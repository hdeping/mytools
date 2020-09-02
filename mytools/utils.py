#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-09-02 10:52:59
    @project      : my toolbox
    @version      : 1.0
    @source file  : utils.py

============================
"""
import os
import numpy as np

class GetDoc():
    """
    docstring for GetDoc
    convert the docstrings into  a README file
    """
    def __init__(self):
        """
        self.modules:
            It is a string array
            module name and their members
            such as ["mytools","DrawCurve"...]
            The first item should the module name,
            and the others are the members.
        self.outputName:
            It is the name for output file,
            initialized by "api.md"
        """
        super(GetDoc, self).__init__()
        
        self.modules = ["mytools",
                        "DrawCurve",
                        "DrawPig",
                        "Excel",
                        "MyCommon",
                        "NameAll",
                        "Triangle",
                        "TurtlePlay"]
        self.ouputName = "api.md"

    def setOutputName(self,outputName): 
        """
        reset self.outputName
        """
        print("output name is set to ",outputName)
        
        self.ouputName = outputName
        return

    def getNewModule(self):
        """
        It is generater for the module name
        for example, mytools --> mytools
                    Triangle --> mytools.Triangle
        """
        for i in range(len(self.modules)):
            name = self.modules[i]
            if i == 0: 
                yield name
            else:
                name = "%s.%s"%(self.modules[0],name)
                yield name

        return
    def output(self):
        """
        convert the docstrings into a txt file with 
        the command pydoc
        """
        for name in self.getNewModule():
            if name == self.modules[0]:
                command = "pydoc %s > %s"%(name,self.ouputName)
            else:
                command = "pydoc %s >> %s"%(name,self.ouputName)
            print(command)
            os.system(command)
            
            
        return

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




