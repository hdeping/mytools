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
import yaml


class MyCommon():
    """docstring for MyCommon
    In this module, I pack some frequenty used functions
    """
    def __init__(self):
        super(MyCommon, self).__init__()
    def setDirs(self,dirs):
        """
        setup for the data directory
        """
        self.dirs = dirs
        return
    def setFilename(self,filename):
        """
        setup for the filename with relative path
        input: filename, such as "data.txt"
        """
        self.filename = self.dirs + filename
        return
    def setFileDirs(self,filename):
        """
        setup for the filename with absolute path
        input: filename, such as "/home/test/data.txt"
        """
        self.filename = filename
        return
    def getCommon(self, dict1, dict2):
        """
        get the common key of the two dicts
        and get a dicts with the difference value
        as a new value.
        """
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
        """
        write dicts data into a json file
        input: data, dicts type
               filename, string type

        """
        print("write to file ",filename)
        fp = open(filename,"w")
        json.dump(data,fp,indent=4,ensure_ascii=False)
        fp.close()

        return
    def loadJson(self, filename):
        """
        load data from the json file
        input: filename, string type
        return: data, dicts type
        """
        print("load data from file ",filename)
        fp = open(filename,"r")
        data = json.load(fp)
        fp.close()

        return data

    def loadFile(self, filename):
        """
        load data from the file
        If it is a yml or yaml type,
        yaml module would be used,
        if it is a json type, json module
        would be used, any other formats are not 
        supported
        input: filename, string type
        return: data, dicts type
        """
        print("load data from file ",filename)
        fp = open(filename,"r")
        if filename.endswith("json"):
            data = json.load(fp)
        elif filename.endswith("yml") or filename.endswith("yaml"):
            data = yaml.load(fp)
        else:
            data = None 
            print("format wrong with ",filename)
            
        fp.close()

        return data

    def writeFile(self, data, filename):
        """
        write data to the file.
        If it is a yml or yaml type,
        yaml module would be used,
        if it is a json type, json module
        would be used, any other formats are not 
        supported.
        write dicts data into a json file
        input: data, dicts type
               filename, string type

        """
        print("write to file ",filename)
        fp = open(filename,"w")
        if filename.endswith("json"):
            json.dump(data,fp,indent=4,ensure_ascii=False)
        elif filename.endswith("yml") or filename.endswith("yaml"):
            yaml.dump(data,fp)
        else:
            data = None 
            print("format wrong with ",filename)
        
        fp.close()

        return
     
    def writeCSV(self, table_array, filename):
        """
        把数组写成csv文件输出，以tab符作为间隔符号
        同时，应去除原始文本中多余的tab符

        write the array into a csv file with the tab as 
        delimiters. On the same time, extraordinary tabs 
        are deliminated.
        """
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
    

