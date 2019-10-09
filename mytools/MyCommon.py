#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-08 10:18:12
    @project      : Some Common Functions Collections
    @version      : 1.0
    @source file  : MyCommon.py

============================
"""
import json

# my common functions
class MyCommon():
    """docstring for MyCommon"""
    def __init__(self):
        super(MyCommon, self).__init__()
    # setup for the data directory
    def setDirs(self,dirs):
        self.dirs = dirs
        return
    # setup for the filename with relative path
    def setFilename(self,filename):
        self.filename = self.dirs + filename
        return
    # setup for the filename with absolute path
    def setFileDirs(self,filename):
        self.filename = filename
        return
    # get the common key of the two dicts
    # and get a dicts with the difference value
    # as a new value.
    def getCommon(self, dict1, dict2):
        print("length 1: ",len(dict1))
        print("length 2: ",len(dict2))

        common_length = 0
        res = {}
        for key in dict1:
            if key in dict2:
                res[key] = dict2[key] - dict1[key]
                common_length += 1
        print("common length: ",common_length)
        return res
    def writeJson(self, data, filename):
        print("write to file ",filename)
        fp = open(filename,"w")
        json.dump(data,fp,indent=4)
        fp.close()

        return
    # load data from the json file
    def loadJson(self, filename):
        print("load data from file ",filename)
        fp = open(filename,"r")
        data = json.load(fp)
        fp.close()

        return data
    # 把数组写成csv文件输出，以tab符作为间隔符号
    # 同时，应去除原始文本中多余的tab符
    def writeCSV(self, table_array, filename):

        with open(filename,'w') as fp:
            # get the csv files with '\t' as seperator

            for i in range(len(table_array)):
                line = table_array[i]
                # print(i)
                for ii in range(len(line) - 1):
                    item = line[ii]
                    item = str(item)
                    item = item.replace('\t','    ')
                    fp.write(str(item)+'\t')
                fp.write(str(line[-1]+'\n'))

        return
    

