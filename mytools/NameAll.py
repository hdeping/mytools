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
        name = re.sub(r"[ .,#?\\{}()\[\]@*#&%!^]",'',filename)
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
        output = output.replace("z-liborg","")
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
                
