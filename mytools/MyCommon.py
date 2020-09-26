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
import os
from tqdm import tqdm
import sys
import urllib
from bs4 import BeautifulSoup
import sqlite3
import datetime

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
    def loadStrings(self,filename,encoding="ISO-8859-2"):
        """
        input: 
            filename, text
        return:
            data, array type, strings line by line
        """
        fp = open(filename,'r',encoding=encoding)
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
        return 
    def loadHtml(self,filename,encoding="ISO-8859-2"):
        """
        docstring for loadHtml
        """
        print("load data from",filename)
        import scrapy
        data = open(filename,"r",encoding=encoding)
        data = data.read()
        res = scrapy.Selector(text=data)

        return res

    def getEarthRadius(self,alpha):
        """
        docstring for getEarthRadius
        the radius is different in the different
        region in the earth

        Earth Radius by Latitude (WGS 84)

        """
        a = 6378137 
        b = 6356752
        A = np.cos(alpha)**2
        B = np.sin(alpha)**2
        radius = (A*a**4+B*b**4)/(A*a**2+B*b**2)
        radius = np.sqrt(radius)
        return radius

    def getGPSCoor(self,arr):
        """
        docstring for getGPSCoor
        arr = [latitude,longtitue,altitude]
        latitude:-90~90
        longtitude: -180~180
        altitude: usually greater than 0
        经纬度数组
        """
        # the radius of the earth
        alpha = arr[0]*np.pi/180 
        theta = arr[1]*np.pi/180
        radius = self.getEarthRadius(alpha)
        radius += arr[2]
        z = np.sin(alpha)*radius
        x = np.cos(alpha)*np.cos(theta)*radius
        y = np.cos(alpha)*np.sin(theta)*radius

        return np.array([x,y,z])

    def getGPSDist(self,arr1,arr2):
        """
        docstring for getGPSDist
        """
        coor1 = self.getGPSCoor(arr1)
        coor2 = self.getGPSCoor(arr2)
        return np.linalg.norm(coor1 - coor2)

    def getSeconds(self,time_string):
        """
        docstring for getSeconds
        2020-08-17T00:02:47.000Z -> seconds
        """
        try:
            time_string = time_string.split("T")[1]
            time_string = time_string.split(".")[0]
            time_string = time_string.split(":")     
            seconds = 0
            for x in time_string:
                seconds = 60*seconds + int(x)
        except Exception as e:
            print("wrong with the time: ",time_string)
            return -1

        return seconds
    def dealGPS(self,filename="20200817074537.json"):
        """
        docstring for dealGPS
        """
        res = self.loadStrings(filename)
        data = []
        for line in res:
            line = line.split(",")
            if line[0] == "time":
                continue
            seconds = self.getSeconds(line[0])
            item = [seconds]
            for i in range(1,4):
                item.append(float(line[i]))
            data.append(item)
        data = np.array(data)

        # calculate the time,distance and speed
        format_string = "%.1f 秒, %.1f 米, %.1f km/h"
        output = []
        for i in range(1,len(data)):
            l1 = data[i-1]
            l2 = data[i]
            t = l2[0] - l1[0]
            s = self.getGPSDist(l2[1:],l1[1:])
            line = [t,s,s*3.6/t]
            # print(format_string%(tuple(line)))
            output.append(line)
        print(np.sum(output,axis=0))
        return output

    def getCommand(self,command):
        """
        docstring for getCommand
        command: system command
        return: the system output
        """
        results = os.popen(command).read()
        results = results.split("\n")
        if results[-1] == "":
            results.pop()
        return results

    def getCurrentImages(self):
        """
        docstring for getCurrentImages
        """
        images = []
        img_types = ["jpg","png","jpeg","gif"]
        for img in img_types:
            command = "ls *" + img 
            images += self.getCommand(command)
        return images

    def writeImageJs(self):
        """
        docstring for writeImageJs
        """
        images = self.getCurrentImages()
        filename = "images.js"
        self.writeJs(images,"images",filename)

        return
    def writeJs(self,data,title,filename):
        """
        docstring for writeJs
        """
        self.writeJson(data,filename)
        with open(filename,'r') as fp:
            data = fp.read()
        with open(filename,'w') as fp:
            fp.write("var %s = %s;"%(title,data))
        return
    def getMyShell(self):
        """
        docstring for getMyShell
        """
        dirs = a.getCommand("ls ../myshell/*")
        strings = ""
        for dir in dirs:
            str1 = "# ------ %s --------\n"%(dir.split("/")[-1])
            str2 = "\n".join(a.loadStrings(dir)) + "\n"
            strings +=  str1 + str2 
        with open("myshell.sh","w") as fp:
            fp.write(strings)

        return  
    def source2Js(self,suffix=["kt"]):
        """
        docstring for source2Js
        """
        lists = self.getCommand("find . -type f")

        sources = {
            "keys":[],
            "values":[]
        }
        for line in tqdm(lists):
            suf = line.split(".")[-1]
            if suf in suffix:
                code = self.loadStrings(line)
                sources["keys"].append(line)
                sources["values"].append("\n".join(code))

        self.writeJs(sources,"sources","sources.js")

    def getAllSources(self,filename="lists.txt"):
        """
        docstring for getAllSources
        """
        data = self.loadStrings(filename)
        types = ['cpp', 'py', 'java', 'h', 'cuh', 'cu', 'cc', 'c', 'hpp']
        results = ""
        for line in tqdm(data):
            suffix = line.split(".")[-1]
            if suffix in types:
                with open(line,"r") as fp:
                    results += "%s\n%s\n"%(line,fp.read())
        filename = "sources.txt"
        with open(filename,'w') as fp:
            fp.write(results)

        return
    def getCode(self):
        """
        docstring for getCode
        get the source code under 
        the current directory
        """
        code_types = {
            "c":["c","h"],
            "cpp":["cpp","hpp","c","h","cc"],
            "cu":["cu","cuh"]
        }
        try:
            types = []
            for item in sys.argv[1:]:
                if item in code_types:
                    types += code_types[item]
                else:
                    types.append(item)
            print(types)
            self.source2Js(suffix=types)
        except Exception:
            print("wrong with input")
        return
    def getMovies(self,filename="xie"):
        """
        docstring for getMovies
        """
        data = a.loadStrings("xie",encoding="utf-8")

        movies = {}
        for i,line in enumerate(data):
            if len(line) == 3 and line[1] == ".":
                movies[data[i-1]] = float(line)
        movies = a.sortDicts(movies)
        for key in movies:
            print(key,movies[key])

        return
    def resize(self,size=30):
        """
        docstring for run
        """
        if len(sys.argv) < 2:
            print("Please input image files")
            print("such as: resize *.jpg *.png")
            return 
        names = sys.argv[1:]
        for i,name in enumerate(names):
            string = name.split(".")[0]
            formats = "convert -resize %d%s %s %s_resize%d.jpg"
            command = formats%(size,"%",name,string,size)
            print(i+1,command)
            os.system(command)
    def getHtmlSoup(self,url):
        """
        docstring for getHtmlSoup
        """
        html = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(html,"lxml")
        
        return soup
    def saveContents(self,contents,filename):
        """
        docstring for saveContents
        contents: string
        filename: name of the output file
        """

        with open(filename,'w') as fp:
            fp.write(contents)

        return
    def connectDB(self,db_name="./sqlite3.db"):
        """
        docstring for connectDB
        initialize self.db and self.conn
        """
        self.db = sqlite3.connect(db_name)
        print("opened %s successfully!"%db_name)
        self.conn = self.db.cursor()
        return
    def exitDB(self):
        """
        docstring for exitDB
        """
        self.db.commit()
        self.db.execute("vacuum;")
        self.conn.close()
        self.db.close()
        return
    def getDate(self):
        """
        docstring for getData
        """
        formats = '{0:%Y-%m-%d %H:%M:%S}'
        date = datetime.datetime.now()
        return formats.format(date)
    def createTable(self,table_name,fields):
        """
        docstring for createTable
        """
        command = "create table if not exists %s(%s);"
        command = command%(table_name,",".join(fields))
        self.conn.execute(command)
        return
    
