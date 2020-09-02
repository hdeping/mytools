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
import re 
import sys

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

class MySort():
    """
    my practice for sorting algorithms """
    def __init__(self):
        super(MySort, self).__init__()

        return

    def mergeSort(self,arr):
        """
        docstring for mergeSort
        arr:
            array 
        """
        if len(arr) == 1:
            return arr 
        elif len(arr) == 2:
            m,n = arr 
            if m > n:
                m,n = n,m 
            return [m,n]
        else:
            num = len(arr) // 2 
            res = []

            arr1 = self.mergeSort(arr[:num])
            arr2 = self.mergeSort(arr[num:])
            count = 0
            # print(arr1,arr2)
            for index,i in enumerate(arr2):
                tmp = arr1[count:].copy()
                for j in tmp:
                    if i < j:
                        res.append(i)
                        break
                    else:
                        res.append(j)
                        count += 1
                if count == num:
                    break
            if count == num:
                res += arr2[index:]
            else:
                res += arr1[count:]

            return res
    def test(self):
        """
        docstring for test
        """
        arr = [3,5,9,1,10,200,2000,-9,10,200,
               2000,-9,20,50,60,55,11]
        # arr = [3,5,9,1,10,200,2000,-9,20,50]
        # arr = [3,5,9,1]
        # arr = [3,9,7]
        print(arr)
        arr = self.mergeSort(arr)
        print(arr)

        return 

import sys
from PyPDF2 import PdfFileReader
from PyPDF2 import PdfFileWriter
from PyPDF2 import PdfFileMerger

class MyPdf():
    """
    dealing with pdf files with PyPDF2
    """
    def __init__(self):
        super(MyPdf, self).__init__()
    def getPages(self):
        """
        docstring for getPages
        get all the pages of the input pdf files
        from the command line
        """  
        names = sys.argv
        num = len(names)
        pages = 0
        count = 0
        for j in range(1,num):
            try:
                input = PdfFileReader(open(names[j],"rb"))
                i = input.numPages
                print("file ",names[j])
                print("there are",i,"pages")
                pages += i
                count += 1
            except Exception:
                print("something is wrong with",names[j])
        print("Total pages are: %d pages"%(pages))
        print("number: %d/%d"%(count,num-1))
        return

    def mergePdfs(self,filenames,mergedName,passwords=None):
        """
        docstring for mergePdfs
        filenames:
            1d array of filenames
        mergedName:
            name of the output file
        passwords:
            passwords for the pdf files if is needed
        """
        # number of the files
        num = len(filenames)
        # note that False should be used
        pdf_merger = PdfFileMerger()
    
        for i in range(num):
            print("adding ",filenames[i])
            fp =  open(filenames[i],"rb") 
            pdf_reader = PdfFileReader(fp,strict=False)
            if not pdf_reader:
                return
            pdf_merger.append(pdf_reader)
            fp.close()

        with open(mergedName, 'wb') as fp:
            print("output to ",mergedName)
            pdf_merger.write(fp)

        return
    def pick100(self,threshold=100):
        """
        docstring for getPages
        get all the pages of the input pdf files
        with over 100 pages from the command line
        """  
        import os 

        names = os.listdir()
        num = len(names)
        pages = 0
        count = 0
        for j in range(1,num):
            try:
                input = PdfFileReader(open(names[j],"rb"))
                i = input.numPages
                if i > threshold:
                    print("%s, %d pages"%(names[j],i))
                    count += 1
            except Exception:
                print("something is wrong with",names[j])
        print("there are %d books over %d pages"%(count,threshold))
        return

class NameAll():
    """docstring for Rename
    rename the files in the current directory
    """
    def __init__(self):
        super(NameAll, self).__init__()
    
    def getCurrentFiles(self):
        """
        get the filenames in the current directory,
        with the help of ls command
        """
        filenames = os.popen("ls")
        filenames = filenames.read()
        filenames = filenames.split('\n')
        # get rid of ""
        filenames.pop()

        return filenames

    def getSuffixIndex(self,filename):
        """
        get the suffix of a string
        such as : foo.xxx -> -3
        filename[-3:] = "xxx"
        """
        # print("------ %s -------"%(filename))
        num = len(filename)
        if num > 7:
            num = 7
        for i in range(1,num):
            if filename[-i] == '.':
                res = 1 - i 
                return res
        return -1

    def arr2string(self,arr):
        """
        array to string
        ["a",'b','c'] -> 'abc'
        one line is enough 
        string = "".join(arr)
        """
        # string = ""
        # for word in arr:
        #     string = string + word
        string = "".join(arr)
        return string 

    def getStdStr(self,filename):
        """
        get a standard string
        "ab ac ab" -> "AbAcAb"
        input: filename, string type
        return: a new string
        """
        # get the capital form of a string array
        # ["a",'b','c'] -> ["A",'B','C']
        filename = self.getCapitalize(filename)
        filename = self.arr2string(filename)
        return filename

    def normallize(self,name):
        """
        get the capitalized format of a word
        """
        return name.capitalize()

    def getCapitalize(self,filename):
        """
        get the capitalized format of a string array
        map function is used here
        filename = list(map(self.normallize,filename))
        """   
        filename = list(map(self.normallize,filename))
        return filename

    def getNewFilename(self,filename):
        """
        get the new filename of a old one,
        characters like [?()[]'\"{}#&/\\,. would
        be deliminated
        "a .. ? b ..pdf" -> "AB.pdf"
        """
        suffix_start = self.getSuffixIndex(filename)
        # get rid of the redundant characters
        #name = re.sub(r"[?()[]'\"{}#&/\\,.]",'',filename)
        name = re.sub(r"[:;'.,#?\\{}()\[\]@*#&%!^]",'',filename)
        new_name   = name.split(' ')
        # if there is no strange characters
        p1 = (len(filename) - len(new_name[0]) == 1) 
        p2 = (filename == new_name[0])
        if  p1 or p2: 
            return filename
        # split the words with ' '
        
        output = self.getStdStr(new_name)
        if suffix_start != -1:
            output = "%s.%s"%(output[:suffix_start],new_name[-1][suffix_start:])
        # get rif of z-lib.org
        output = output.replace("Z-liborg","")
        # new_good_times.pdf --> NewGoodTimes.pdf
        if "_" in output:
            output = output.split("_")
            output = self.getStdStr(output)
        return output
        
    def run(self):
        """
        rename the files with the help of mv 
        command after we get the new names
        """
        filenames = self.getCurrentFiles() 
        print(filenames)

        count = 0
        for name in filenames:
            new_name = self.getNewFilename(name)
            if name != new_name:
                count += 1
                command = "mv '%s' %s"%(name,new_name)
                print(count,command)
                print("running the rename program")
                os.system(command)

        return 

class OpenFiles():
    """
    open files with different commands
    """
    def __init__(self):
        """
        self.video_types:
            filename extension for video and audio
            types,array
        self.image_types:
            filename extensions for image types,array
        self.text_types:
            filename extensions for text types,array
        self.other_types:
            filename extensions and operation commands,
            dictionary
        """
        super(OpenFiles, self).__init__()
        self.video_types = ["mp4","avi","rmvb","webm","ts",
                            "mp3","ogg","wav","flac","mov"]
        self.image_types = ["jpg","jpeg","gif","png","bmp","icon"]
        self.text_types  = ["txt","py","c","h","html","css","js","gh",
                            "lisp","cpp","go","f","f90",
                            "java","pl","log","tex","bbl","aux",
                            "bib","sh","php","makefile","Makefile",
                            "rst","config","gitconfig"]
        self.other_types = {"md":"typora",
                            "pdf":"evince",
                            "ps":"evince",
                            "docx":"wps",
                            "ppt":"wpp",
                            "pptx":"wpp",
                            "xls":"et",
                            "xlsx":"et",
                            "doc":"wps",
                            "lyx":"lyx",
                            "blend":"blender"
                            }

    def runCommand(self,program,i):
        """
        open the i-th file withe a specific program
        """
        command = "%s '%s'"%(program,sys.argv[i])
        os.system(command)
        return

    def getSuffix(self,arg):
        """
        get the suffix of a path
        for example:
            main.py --> py
            .git/main.py --> py
            dir/main.py --> py
        """
        arg = arg.split("/")
        arg = arg[-1]
        arg = arg.split(".")
        arg = arg[-1]
        return arg 
        
    def run(self,i):
        """
        input: i, index number of the command parameters
        return: None, filename extensions will be classified 
                and open with the corresponding command
        """
        arg = self.getSuffix(sys.argv[i])
        if   arg in self.video_types:
            self.runCommand("mplayer",i)
        elif arg in self.text_types:
            self.runCommand("subl3",i)
        elif arg in self.image_types:
            self.runCommand("eog",i)
        elif arg in self.other_types:
            command = self.other_types[arg]
            self.runCommand(command,i)
        else:
            print("unknown type: " + arg)

        return

    def main(self):
        """
        main function for analyzing
        each command parameters
        """
        if len(sys.argv) == 1:
            print("please input a file")
        else:
            for i in range(1,len(sys.argv)):
                self.run(i)

        return

class RunCommand():
    """
    run the shell command with os.system
    """
    def __init__(self):
        super(RunCommand, self).__init__()
    def gitRebase(self,num):
        """
        docstring for gitRebase
        input: 
            num, an integer number, last commit numbers
        return:
            None, but the command git rebase -i HEAD~num
            was executed
        """
        command = "git rebase -i HEAD~%d"%(num)
        print(command)
        os.system(command)
        return
    def runGitRebase(self):
        """
        docstring for runGitRebase
        run self.gitRebase accepted 
        a argument from the command line
        """
        try:
            num = int(sys.argv[1])
            self.gitRebase(num)
        except IndexError:
            print("you need a command line argument")
        return
        
