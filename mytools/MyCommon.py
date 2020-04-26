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
import numpy as np


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
    def getStatiNumber(self,dicts):
        """
        distill the number from the ingredients array
        :param ingredients:
            array type, all the ingredients
        :return:
            dicts type, ingredients and their number
        """
        results = {}
        for line in dicts:
            if line in results:
                results[line] += 1
            else:
                results[line] = 1

        return results
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
    # 从json文件中读取数据
    def loadJson(self, filename,encoding="utf-8"):
        """
        load data from the json file
        input: filename, string type
        return: data, dicts type
        """
        print("load data from file ",filename)
        fp = open(filename,"r",encoding=encoding)
        data = fp.read()
        if data.startswith(u'\ufeff'):
            data = data.encode('utf8')[3:].decode('utf8')
        data = json.loads(data)

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
        input: data, dicts type or list type
               filename, string type

        """
        print("write to file ",filename)
        fp = open(filename,"w")
        if filename.endswith("json"):
            json.dump(data,fp,indent=4,ensure_ascii=False)
        elif filename.endswith("yml") or filename.endswith("yaml"):
            yaml.dump(data,fp)
        else:
            for line in data:
                fp.write("%s\n"%(line))
        
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
    def loadStrings(self,filename):
        """
        input: 
            filename, text
        return:
            data, array type, strings line by line
        """
        fp = open(filename,'r')
        data = fp.read()
        fp.close()
        data = data.split("\n")
        if data[-1] == "":
            data.pop()
        return data        
        
    def getStringStati(self, array):
        """
        get the statistic information of a string array
        input:
            array, string array
        return:
            result, dicts type, such as {"aaa":100,"bbb":10,...}

        """
        result = {}
        for line in array:
            if line in result:
                result[line] += 1
            else:
                result[line] = 1
        return result

    def sortDicts(self,dicts):
        """
        sort the input dicts
        input:
            dicts, with number as values,
            such as {"aaa":100,...}
        output:
            dicts, sorted dicts, the first value should
            be a maximum
        """
        keys, values = self.dicts2KeyValue(dicts)
        
        indeces = np.argsort(values)
        # small --> big transformed to
        # big   --> small
        indeces = np.flip(indeces)
        keys    = keys[indeces]
        values  = values[indeces]
        result  = self.keyValue2Dicts(keys,values)
        
        return result

        
    def keyValue2Dicts(self,keys,values):
        """
        input:
            keys and values with numpy array type,
            which should be the same shape
        return:
            dicts
        """
        assert len(keys) == len(values)
        result = {}
        for key,value in zip(keys,values):
            result[key] = value
        return result
    def dicts2KeyValue(self,dicts):
        """
        input:
            dicts
        return:
            keys and values with numpy array type
        """
        keys   = []
        values = []
        for key in dicts:
            value = dicts[key]
            keys.append(key)
            values.append(value)

        keys    = np.array(keys)
        values  = np.array(values)
        return keys,values
    def printDicts(self,dicts,num):
        """
        input:
            dicts: dicts type
            num: interger type, print first num elements
        return:
            None
        """
        for i,key in enumerate(dicts):
            value = dicts[key]
            print(i,key,value)
            if i > num - 1:
                break
        pass


