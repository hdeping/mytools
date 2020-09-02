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
import tkinter
from tkinter import messagebox
import tkinter.ttk as ttk
from .Triangle import Triangle

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

class MyGUI(Triangle):
    """
    my gui based on tkinter
    """
    def __init__(self):
        """
        self.window:
            the tkinter window
        self.height:
            height of the widget
        self.width:
            width of the widget
        """
        super(MyGUI, self).__init__()
        self.window = tkinter.Tk()
        self.window.title("三角形面积计算器")
        self.window.geometry("480x480")
        self.tkLengths = []
        for i in range(3):
            self.tkLengths.append(tkinter.StringVar())

        self.width = 17
        self.height = 5

    def setWidgetSize(self,width,height):
        """
        setup for the height and the width 
        of the tkinter window
        """

        self.width  = width
        self.height = height

        return
        
    def interface(self):
        """
        main function for interface design
        """
        texts = ["一","二","三"]
        prefix = "请输入边长"
        for i in range(3):
            self.setLabel(prefix+texts[i],i,0)
            self.setEntry(i,i,1)
        self.setButton()
        # self.width = self.width*3
        # self.height = self.height*3
        # self.setLabel("结果",3,0)
        self.setText()

        self.window.mainloop()
        return
    def setText(self):
        """
        set text box
        """
        self.text = tkinter.Text(self.window,
                            background="lightblue",
                            width=self.width,
                            height=self.height)
        self.text.grid(row=3,column=1)
    def setLabel(self,text,row,col):
        """
        function for label settings
        """
        self.label = tkinter.Label(self.window,
                                   text=text,
                                   background="lightgreen",
                                   width=self.width,
                                   height=self.height)
        if row == 3:
            self.label.grid(columnspan=3,sticky = tkinter.S)
        else:
            self.label.grid(row=row, column=col,
                            sticky=tkinter.N + tkinter.S)
        return
    def setEntry(self,index,row,col):
        """
        function for entry settings
        """
        entry = tkinter.Entry(self.window,
                              textvariable=self.tkLengths[index],
                              width=self.width)
        entry.grid(row=row,column=col,
                   sticky=tkinter.N + tkinter.S)

        return
    def setButton(self,text=None):
        """
        function for button settings
        """
        button = tkinter.Button(self.window,
                                text = "计算面积",
                                width=self.width,
                                height=self.height,
                                foreground="blue",
                                command=self.show)
        button.grid(row=0,column=2,
                    sticky=tkinter.N + tkinter.S,
                    rowspan = 3,
                    columnspan = 2)
        return
    def show(self):
        """
        binding function for the command
        area is calculated and displayed 
        in the text box
        """
        a = float(self.tkLengths[0].get())
        b = float(self.tkLengths[1].get())
        c = float(self.tkLengths[2].get())
        self.setLengths(a,b,c)
        if self.isTriangle():
            self.getArea()
            text = "area is %.2f\n"%(self.area)
            # text = "triangle (%.2f,%.2f,%.2f), area is %.2f"%(a,b,c,area)
            # self.label.configure(text=text)
            self.text.insert(tkinter.END,text)
        else:
            text = "%.2f,%.2f,%.2f cannot construct a triangle"%(a,b,c)
            messagebox.showinfo("Error",text)

        return


