#!/usr/local/bin/python3
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-04-10 17:27:39
                    2019-10-07 11:14:02
    @project      : rename all the files with blanks
    @version      : 1.0
    @source file  : nameall.py

============================
"""

import os
import sys

import re

# rename the files in the current
# directory
class NameAll():
    """docstring for Rename"""
    def __init__(self):
        super(NameAll, self).__init__()
    
    def getCurrentFiles(self):
        filenames = os.popen("ls")
        filenames = filenames.read()
        filenames = filenames.split('\n')
        # get rid of ""
        filenames.pop()

        return filenames

    # get the suffix of a string
    # such as : foo.xxx -> -3
    # filename[-3:] = "xxx"
    def getSuffixIndex(self,filename):
        # print("------ %s -------"%(filename))
        for i in range(1,7):
            if filename[-i] == '.':
                res = 1 - i 
                return res
        return -1

    # array to string
    # ["a",'b','c'] -> 'abc'
    def arr2string(self,arr):
        # string = ""
        # for word in arr:
        #     string = string + word
        string = "".join(arr)
        return string 

    # get standar string
    # "ab ac ab" -> "AbAcAb"
    def getStdStr(self,filename):
        # get the capital form of a string array
        # ["a",'b','c'] -> ["A",'B','C']
        filename = self.getCapitalize(filename)
        filename = self.arr2string(filename)
        return filename

    # get the capitalized format of a word
    def normallize(self,name):
        return name.capitalize()

    # get the capitalized format of a string array
    def getCapitalize(self,filename):   
        filename = list(map(self.normallize,filename))
        return filename

    # "a .. ? b ..pdf" -> "AB.pdf"
    def getNewFilename(self,filename):
        suffix_start = self.getSuffixIndex(filename)
        # get rid of the redundant characters
        #name = re.sub(r"[?()[]'\"{}#&/\\,.]",'',filename)
        name = re.sub(r"[ .,#?\\{}()\[\]@*#&%!^]",'',filename)
        new_name   = name.split(' ')
        p1 = (len(filename) - len(new_name[0]) == 1) 
        p2 = (filename == new_name[0])
        if  p1 or p2: 
            return filename
        # split the words with ' '
        
        output = self.getStdStr(new_name)
        output = "%s.%s"%(output[:suffix_start],new_name[-1][suffix_start:])
        return output
        
    def run(self):
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
                
