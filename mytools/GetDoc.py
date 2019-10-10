#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-09 10:39:12
    @project      : get the docstring of a module
    @version      : 1.0
    @source file  : GetDoc.py

============================
"""
import os

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
