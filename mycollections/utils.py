#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-09-02 09:51:47
    @project      : my utils collection
    @version      : 1.0
    @source file  : utils.py

============================
"""
import res
import numpy as np
import json
from mytools import MyCommon

class Ris2Bib():
    """
    transfer .ris file to a .bib one
    """
    def __init__(self):
        """
        self.risfile:
            name of the .ris file
        self.ris2Bib:
            dicts, keywords of .ris to .bib file
        self.bibValues:
            dicts, information of the document,
            author, title, journal, volume, pages,
            year, doi, publisher, abstract, url
        """
        super(Ris2Bib, self).__init__()
        self.getRisfile()
            
        self.ris2Bib = {
            "TY" : "JOUR",
            "AU" : "author",
            "TI" : "title",
            "JA" : "journal",
            "JO" : "journal",
            "VL" : "volume",
            "PY" : "year",
            "SP" : "startingpage",
            "EP" : "finalpage",
            "L3" : "doi",
            "DO" : "doi",
            "PB" : "publisher",
            "AB" : "abstract",
            "UR" : "url",
            "DA" : "date"
        }
        self.bibValues = {
            "author"       : [],
            "title"        : None,
            "journal"      : None,
            "volume"       : None,
            "year"         : None,
            "pages"        : [None,None],
            "doi"          : None, 
            "publisher"    : None,
            "abstract"     : None,
            "url"          : None
        }

        return

    def getRisfile(self):
        """
        docstring for getRisfile
        get the risfile from the command line input
        """
        try:
            self.risfile       = sys.argv[1]
        except IndexError:
            print("Please input a file !")
            print("such as: python3 Ris2Bib.py foo.ris")
            sys.exit()
        return
    def setFilename(self,filename):
        """
        docstring for setFilename
        set the risfile from the method 
        argument
        """
        self.risfile = filename
        return
    def getDocInfo(self):
        """
        docstring for getDocInfo
        get all the informations of the document
        by reading a ris file
        """
        
        with open(self.risfile,'r') as fp:
            for data in fp:
                data = data.split('-',1) 
                if len(data)==1:
                    pass
                else:
                    field = data[0].strip(' ')
                    try:
                        field = self.ris2Bib[field]
                    except KeyError:
                        print("There is no ", field)
                    value = data[1].strip(' ').strip('\n').strip('\r')
                    #print case
                    if field == 'author':
                        self.bibValues[field].append(value)
                    elif field == 'year':
                        value = value.rsplit('/')[0]
                        self.bibValues[field] = value
                    elif field == 'startingpage':
                        self.bibValues["pages"][0] = value
                    elif field == 'finalpage':
                        self.bibValues["pages"][1] = value
                    elif field in self.bibValues:
                        self.bibValues[field] = value
                    
        return
    def tranDocInfo(self):
        """
        docstring for tranDocInfo
        return: 
            lines, string array, contents of  the bib file
        """
        # dealing with the data
        lines=[]
        for key in self.bibValues:
            value = self.bibValues[key]
            if key == "author":
                firstauthor = value[0].rsplit(',')[0].strip(' ')
                name = (firstauthor.lower(),self.bibValues["year"])
                lines.append('@article{%s%s,' % name)
                authors   = ' and '.join(value)
                authorline = "    author = {%s}," % authors 
                lines.append(authorline)
            elif key == "pages":
                if value[0] is not None and value[1] is not None:
                    value = tuple(value)
                    line  = "    pages = {%s--%s}," % value
                    lines.append(line)
            else:
                if value is not None:
                    line = "    %s = {%s}," % (key,value)
                    lines.append(line)
            
        lines.append('}\n')
        return lines
    def printLines(self,lines):
        """
        docstring for printLines
        print the contents of bib file out
        """     
        bibfile = self.risfile[:-4] + ".bib"
        print('Writing output to file ',bibfile)
        with open(bibfile,'w') as fp:
            fp.write('\n'.join(lines))
        return

    def run(self):
        """
        docstring for run
        get the information of the document
        and transform them into a bib format.
        Finally, write the data into a file
        """
        self.getDocInfo()
        lines = self.tranDocInfo()
        self.printLines(lines)
        
        return
        

class TianGanDiZhi(MyCommon):
    """
    天干地支
    """
    def __init__(self):
        """
        self.tiangan:
            10 elements
        self.dizhi:
            12 elements
        self.years:
            ["甲子","乙丑"...]
        self.dicts:
            {"甲子":1984,...}
        self.titles:
            年号和皇帝
        """
        super(TianGanDiZhi, self).__init__()
        self.tiangan = ["甲","乙","丙","丁","戊",
                        "己","庚","辛","壬","癸"]
                        
        self.tiangan = self.tiangan*6
        self.dizhi   = ["子","丑","寅","卯",
                        "辰","巳","午","未",
                        "申","酉","戌","亥"]
        self.dizhi   = self.dizhi*5
        self.getYears()
        self.titles = self.loadJson("titles.json")
        return 
    def getYears(self):
        """
        docstring for getYears
        天干，10
        地支，12
        totally 60
        """
        self.years = []
        for line in zip(self.tiangan,self.dizhi):
            year = "".join(line)
            self.years.append(year)
        # print(self.years)
        dicts = {}
        for i,line in enumerate(self.years):
            dicts[line] = 1984 + i 
        self.dicts = dicts
        
        return
    def getDynasties(self,year):
        """
        docstring for getEmperor
        """
        dynasties = {}
        dynasty   = {}
        dynasty["capital"] = ["阳城","阳翟","斟鄩","商丘","纶城"]
        dynasty["emperors"] = ["帝启","太康","仲康","帝相",
                               "少康","帝抒","帝槐","帝芒",
                               "帝泄","不降","帝扃","胤甲",
                               "帝孔甲","帝皋","帝发","履癸"]
        dynasty["timeline"] = [-2070,-1600]
        dynasties["夏"] = dynasty

        dynasty   = {}
        dynasty["emperors"] = ["商太祖","商代王","商哀王","商懿王",
                               "商太宗","商昭王","商宣王","商敬王",
                               "商元王","商中宗","商孝成王","商思王",
                               "商前平王","商穆王","商桓王","商僖王",
                               "商庄王","商顷王","商悼王","商世祖",
                               "商章王","商惠王","商高宗","商后平王",
                               "商世宗","商甲宗","商康祖","商武祖",
                               "商匡王","商德王","商纣王"]
        dynasty["names"] = ["商汤","太乙","外丙","仲壬","太甲","沃丁",
                            "太庚","小甲","雍己","太戊","仲丁","外壬",
                            "河亶甲","祖乙","祖辛","沃甲","祖丁","南庚",
                            "阳甲","盘庚","小辛","小乙","武丁","祖庚",
                            "祖甲","廪辛","庚丁","武乙","文丁","帝乙",
                            "帝辛"]


        dynasty["titles"] = ["子天乙","子-","子胜","子庸","子至","子绚",
                             "子辩","子高","子密","子伷","子庄","子发",
                             "子整","子滕","子旦","子逾","子新","子更",
                             "子和","子旬","子颂","子敛","子昭","子跃",
                             "子载","子先","子嚣","子瞿","子托","子羡",
                             "子寿"]
        dynasty["timeline"] = [-1600,-1046]
        dynasty["capital"]  = ["镐京","洛阳"]
        dynasties["商"] = dynasty

        dynasty   = {}
        dynasty["emperors"] = ["周武王","周成王","周康王","周昭王",
                               "周穆王","周共王","周懿王","周孝王",
                               "周夷王","周厉王","周宣王","周幽王"]
        dynasty["names"]    = ["姬发","姬诵","姬钊","姬瑖","姬满",
                               "姬紧扈","姬囏","姬辟方","姬变",
                               "姬胡","姬靖","姬宫湼"]
        dynasty["timeline"] = [-1046,-771]
        dynasty["capital"]  = ["镐京"]
        dynasties["西周"] = dynasty

        dynasty   = {}
        dynasty["emperors"] = ["周本王","周桓王","周庄王","周厘王",
                               "周惠王","周襄王","周顷王","周匡王",
                               "周定王","周简王","周灵王","周景王",
                               "周悼王","周敬王","周元王","周贞定",
                               "周哀王","周思王","周考王","周威烈",
                               "周安王","周烈王","周显王","周慎靓王",
                               "周赧王"] 
        dynasty["names"]    = ["姬宜臼","姬林","姬铊","姬胡齐","姬阆",
                               "姬郑","姬壬臣","姬班","姬瑜","姬夷",
                               "姬泄心","姬贵","姬猛","姬匈","姬仁",
                               "姬王介","姬去疾","姬叔","姬嵬","姬王午",
                               "姬骄","姬喜","姬扁","姬定","姬延"]
        dynasty["timeline"] = [-770,-256]
        dynasty["capital"]  = ["洛阳"]
        dynasties["东周"] = dynasty

        dynasty   = {}
        dynasty["emperors"] = ["始皇帝政","二世胡亥","子婴"]
        dynasty["timeline"] = [-221,-206]
        dynasty["capital"]  = ["咸阳"]
        dynasties["秦"] = dynasty

        # dynasty   = {}
        # dynasty["emperors"] = 
        # dynasty["timeline"] = 
        # dynasty["capital"]  = [""]
        # dynasties[""] = dynasty
        dynasties["lists"] = ["西汉","新朝","东汉","魏","蜀","吴",
                              "西晋","东晋","蜀","前赵","后赵","前燕",
                              "前秦","前凉","后秦","后燕","南燕","北燕",
                              "后凉","南凉","西凉","北凉","西秦","夏",
                              "宋","齐","梁","陈","北魏","东魏",
                              "西魏","北齐","北周","隋朝","唐朝","后梁",
                              "后唐","后晋","后汉","后周","前蜀","吴",
                              "楚","闽","南唐","荆南","南汉","吴越",
                              "北汉","后蜀","北宋","南宋","元朝","明朝",
                              "清朝"]

        print(json.dumps(dynasties,indent=4,ensure_ascii=False))
        
        
        return
    def getReignTitles(self):
        """
        docstring for getReignTitles
        """
        filename = "reignTitles.txt"
        data = self.loadStrings(filename)
        data = np.array(data)
        data = data.reshape((-1,3))
        print(data)
        titles = {}
        for line in data:
            values = line[1].split("~")
            years  = []
            print(line,values)
            years.append(int(values[0]))
            if values[1] == "":
                years.append(int(values[0]))
            else:
                years.append(int(values[1]))
            titles[line[0]] = years
        
        data = self.loadStrings("emperors.txt")
        for line in data:
            # print(line)
            line = line.split(" ")
            print(line)
            years = []
            years.append(int(line[1]))
            if line[2] == "":
                years.append(int(line[1]))
            else:
                years.append(int(line[2]))
            titles[line[0]] = years 

        # print(titles)
        self.writeJson(titles,"titles.json")
        
        return
    def getResults(self,data):
        """
        docstring for getResults
        input:
            data, strings
        """
        labels = ["-","年","月"]
        # 0,1,2,0,1,2,... all the 012cycles
        results = ""
        for i,line in enumerate(data):
            if labels[0] in line:
                line = "1"
            elif labels[1] in line:
                line = "2"
            elif labels[2] in line:
                line = "2"
            else:
                line = "0"
            results += line
            if i%3 == 2:
                results += "\n"


        # print(data)
        print(results)
        return
    def searchTitle(self,year):
        """
        docstring for searchTitle
        """
        for key in self.titles:
            value = self.titles[key]
            if year >= value[0] and year <= value[1]:
                print("%s%d年"%(key,year - value[0] + 1))
        return
    def run(self,key):
        """
        docstring for run
        """
        if key in self.dicts:
            year  = self.dicts[key]
            years = np.arange(year,0,-60)
            print(key,years)
        elif key.isdigit():
            year = int(key) - 1984 
            year = year % 60
            year = self.years[year]
            print("%s: %s 年"%(key,year))
            year = int(key)
            self.searchTitle(year)
            
        else:
            print("%s is invalid"%(key))
   
        
        return
    def test(self):
        """
        docstring for test
        """
        key = '庚戌'
        key = "1756"
        self.run(key)
        return

