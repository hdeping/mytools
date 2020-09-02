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
import re
from xml.etree.ElementTree import parse
from xml.etree.ElementTree import ElementTree
import os
from mytools import DrawCurve
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
from wxpy import *

class AnalyzeTree(object):
    """docstring for AnalyzeTree"""
    def __init__(self):
        super(AnalyzeTree, self).__init__()
    
        self.root = parse('descript.xml')

    def showTree(self,rootNode):
        items = rootNode.findall("./")
        if len(items) == 0:
            print(rootNode.tag,rootNode.text)
        else:
            print("---------")
            if type(rootNode) != ElementTree:
                print(rootNode.tag)

            for item in items:
                self.showTree(item)

        return 

    def test(self):
        self.dicts = []
        self.showTree(self.root)
        packages = self.root.findall("./")[-1]

        return 
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

class Wiki2Txt():
    """
    get rid all kinds of strange characters
    int the wiki contents
    """
    def __init__(self):
        super(Wiki2Txt, self).__init__()
        

        #filename = "wiki_AritificialNeuralNetwork.md"
        filename = "wiki_ANN.md"

        fp = open(filename,'r')
        self.data = fp.read()
        fp.close()
    def getStrings(self):
        """
        docstring for getStrings
        """
        self.strings = {
            u"\\[|\\]"                 : "",
            u"\\{|\\}"                 : "",
            u'"'                       : "",
            u'\*\*.*?\*\*'             : "",
            u"\\(pdf\\)"               : "",
            u"\\(PDF\\)"               : "",
            u"Retrieved \d+-\d+-\d+\." : "",
            u";"                       : ",",
            u"\^"                      : "",
            u"\*"                      : "",
            u" PMID \d+\."             : "",
            u"\n\d+\. "                : "\n"
        }

        return
    def transform(self):
        """
        docstring for transform
        a = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", s)
        website in the parentheses are deleted （）
        brackets are deleted  []
        braces are deleted  {}
        double quotation marks are deleted ""
        characters like **a**, **b** are deleted
        (PDF) (pdf) are deleted
        Retrieved 
        ; to ,
        ^,* are deleted
        """
        self.getStrings()
        output = re.sub(u"\\(http.*?\\)", "", self.data)

        for key in strings:
            value  = self.strings[key]
            output = re.sub(key,value,output)
        return output

    def run(self):
        """
        docstring for run
        run the process
        """
        output = self.transform()
        filename = "new_wiki_ANN.txt"

        fp = open(filename,'w')
        fp.write(output)
        fp.close()

        output = output.split("\n")
        print(len(output))

        #print(year)
        year = np.loadtxt("year",dtype=int)
        #print(year)
        order = np.argsort(year)
        for i,index in enumerate(order):
            #print(i+1,output[index])
            print(output[index])
        return
class Bitcoin(MyCommon,DrawCurve):
    """
    wrapper for bitcoin-cli"""
    def __init__(self):
        """
        self.command:
            command for bitcoin-cli
        """
        super(Bitcoin, self).__init__()
        self.command = "bitcoin-cli getblock `bitcoin-cli getblockhash %d`"
        self.keys    = ["size","weight","height",
                        "time","nonce","nTx","difficulty"]
        # self.blocksNum = 500
        self.blocksNum = 607625
        self.width  = 4
        return

    def getInfo(self,content):
        dicts = {}
        for key in self.keys:
            dicts[key] = content[key]

        dicts["date"] = self.getDate(content["time"])

        return dicts

    def getDate(self,timestamp):
        """
        timestamp: 
            such as 1354216278
        return:
            date 
        """
        timeArray = time.localtime(timestamp)
        date = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)

        return date 
    def run(self):
        results = {}
        t1 = time.time()
        for i in range(self.blocksNum):
            command = self.command % (i)
            content = os.popen(command).read()
            content = json.loads(content)

            key     = content["hash"]
            results[key] = self.getInfo(content)
            if i % 100000 == 0:
                print(i)
                self.writeJson(results,"bitcoin%d.json"%(i))

                
        self.writeJson(results,"bitcoin.json")
        print(time.time() - t1)

        return
    def plotData(self,y,z):
        # print(y)
        freq = 1000
        print("sum of nTx: ",sum(y))
        num = (len(y) // freq)*freq
        y = np.array(y[:num])
        y = y.reshape((-1,freq))
        y = np.average(y,axis=1)
        x = np.arange(len(y))*10*freq/60/24/365
        
        z = np.array(z[:num])
        z = z.reshape((-1,freq))
        z = np.average(z,axis=1)
        # z = np.log10(z)
        # y = np.log10(y)
        plt.figure(figsize=(9,9))
        plt.xlabel("Time/year",fontsize=24)
        plt.ylabel("Difficulty/Transactions",fontsize=24)
        plt.title("Difficulty/Transactions --- Time",fontsize=28)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.semilogy(x,y,lw = self.width,label="Transactions")
        plt.semilogy(x,z,lw = self.width,label="Difficulty")
        plt.legend(loc = "upper left",fontsize = 24)
        plt.savefig("bitcoin.png",dvi=200)
        plt.show()

        return
    def test(self):
        filename = "bitcoin.json"
        data = self.loadJson(filename)
        # print(data)
        y = []
        z = []
        blockTime = []
        for key in data:
            y.append(data[key]["nTx"])
            z.append(data[key]["difficulty"])
            blockTime.append(data[key]["time"])
        self.plotData(y,z)


        blockTime = np.array(blockTime)
        blockTime = blockTime / 60
        print(blockTime)
        blockTime = blockTime[1:] - blockTime[:-1]
        
        #plt.hist(blockTime)
        # plt.show()
        print(blockTime)
        print(min(blockTime))
        print(max(blockTime))
        for i,item in enumerate(blockTime):
            if item < 0:
                print(i,item)
                break
        print("average time",np.average(blockTime))
        sections = np.arange(20)
        cuts = pd.cut(blockTime,sections)
        counts = pd.value_counts(cuts)
        print(counts)
        counts.plot.bar()
        # plt.show()


        return
class SublimeToVim():
    """
    docstring for SublimeToVim
    transform snippets files for vim to 
    ones for sublime
    """
    def __init__(self, directory):
        """
        self.directory:
            directory contained snippets files
        self.filenames:
            snippets files under self.directory
        """
        self.directory = directory
        self.filenames = []
        
        return
    def get_filenames(self):
        """TODO: Docstring for get_filenames.
        :returns: TODO
        get all the snippets files under self.directory
        """
        filenames = os.popen("ls %s/*snippets"%(self.directory))
        filenames = filenames.read().split("\n")
        filenames.pop()
        self.filenames = filenames
        #print(self.filenames)
        return
    def get_results(self):
        """TODO: Docstring for get_results.
        :returns: TODO
        print the all the snippets for sublime
        into ones for vim, the results would be written
        into "%s.snippets"%(self.directory)
        """
        filename = "%s.snippets"%(self.directory)
        print("writing to %s"%(filename))
        fp = open(filename,"w")
        for line in self.filenames:
            print(line,len(line))
            content,field = self.get_content_field(line)
            fp.write("snippet %s\n"%(field))
            fp.write(content+"\n")
            fp.write("endsnippet\n\n")

        fp.close()
        return
        
    def get_content_field(self,filename):
        """
        input:
            filename, snippet file
        return:
            contents, fields, contents and fields for
            each snippet in filename
        """
        # read data
        fp = open(filename,"r")
        data = fp.read().split("\n")
        data.pop()
        fp.close()

        # get the contents
        content_begin_line = []
        content_end_line   = []

        # get all the fields
        fields = []
        for i,line in enumerate(data):
            if "endsnippet" in line:
                content_end_line.append(i)
                

            elif "snippet" in line:
                content_begin_line.append(i+1)
                line = line.split(" ")
                fields.append(line[1])

        contents = []
        assert len(content_begin_line) == len(content_end_line)

        for i in range(len(content_begin_line)):
            res = self.arr_to_string(data,
                            content_begin_line[i],
                            content_end_line[i])
            contents.append(res)
        return contents,fields

    def arr_to_string(self,arr,begin,end):
        """TODO: Docstring for arr_to_string.

        :arr: input a array
        :begin: begin index
        :end: end index pythonix way (0,2)-> 0,1
        :returns: string 

        """
        string = ""
        for i in range(begin,end):
            string += arr[i] + "\n"
        string = string[:-1]
        return string
class VimToSublime(SublimeToVim):

    """
    Docstring for VimToSublime. 
    It is inheritated from SublimeToVim,
    and it is used to transform snippets files for 
    sublime to ones for vim
    """

    def __init__(self):
        """
        TODO: to be defined1. 
        self.filenames:
            filenames of 
        self.dirs:
        self.sources:
        self.texts:
        """
        SublimeToVim.__init__(self,".")
        self.filenames = []
        self.dirs = []
        self.sources = ["c","cpp","cuda",
                       "javascript", "python"]
        self.texts = ["tex"]
    def get_dirs(self):
        """TODO: Docstring for get_dirs.
        :returns: TODO
        get self.dirs
        """
        self.get_filenames()
        for line in self.filenames:
            line = line.split(".")
            line = line[1][1:]
            self.dirs.append(line)

    def test_cuda(self):
        """
        try to get the contents and fields
        under the directory "cuda"
        """
        self.get_dirs()
        name = self.filenames[2]
        contents,fields = self.get_content_field(name)
        print(fields)
        

    def get_all_snippets(self):
        """
        get all the snippets in a snippet file for vim
        """
        self.get_dirs()
        for i,name in enumerate(self.filenames):

            # print(name)
            
            lang = name.split(".")
            lang = lang[1][1:]
            # print(lang)
            
            if lang in self.sources:
                # print("sources",lang)
                scope = "source." + self.dirs[i]
            elif lang in self.texts:
                # print("texts",lang)
                scope = "text." + self.dirs[i]
            else:
                continue                
                
            contents,fields = self.get_content_field(name)
            for content,field in zip(contents,fields):
                if "\t" in field:
                    field = field.replace("\t","")
                if field == "<<":
                    field = "kernel"
                self.write_to_sublime(i, content, field, scope)
        # print(fields)
        return

    def write_to_sublime(self,index,content,field,scope):
        """
        input:
            index, index of self.dirs
            content, content of the snippet
            field, field of the snippet
            scope, language type, such as python, ruby and so on
        return:
            None, but the results should be written into a 
            sublime-snippet file
        """
        if not os.path.exists(self.dirs[index]):
            os.mkdir(self.dirs[index])
        filename = "%s/%s.sublime-snippet"%(self.dirs[index],field)
        fp = open(filename,'w')

        print("writing to %s"%(filename))
        
        fp.write("<snippet>\n")
        fp.write("    <content><![CDATA[\n")
        fp.write("%s\n"%(content))
        fp.write("]]></content>\n")
        fp.write("    <tabTrigger>%s</tabTrigger>\n"%(field))
        fp.write("    <scope>%s</scope>\n"%(scope))
        fp.write("</snippet>\n")
        fp.close()
        return
class Urine():
    """
    docstring for Volume
    lengths were measured by a ruler
    volumes were computed by the formula for 
    the volume of a circular truncated cone
    """
    def __init__(self):
        super(Urine, self).__init__()
        
    def getVolume(self,r,R,l):
        """
        docstring for getVolume
        """
        h = (R-r)**2
        h = np.sqrt(l*l - h)
        volume = np.pi*(R*R + R*r+r*r)*h/3
        return volume

    def getVolume2(self,l1):
        """
        docstring for getVolume
        R1 = (l1*R + l2*r)/(l1+l2)
        """
        r  = 2.5
        l  = 15.6
        R  = 4.3
        l2 = l - l1 
        R1 = (l1*R + l2*r)/l
        volume = self.getVolume(r,R1,l1)
        return volume

    def getRadius(self,a,b,c):
        """
        docstring for getRadius
        """
        p = (a+b+c)/2 
        s = p*(p-a)*(p-b)*(p-c)
        s = np.sqrt(s)
        R = a*b*c/(4*s)
        return R

    def test(self):
        """
        docstring for test
        """

        mine = np.array([144.0,144.0,144.2,144.2,144.0])
        total = np.array([146.1,146.1,146.1,146.3,146.1])

        volume = np.average(total) - np.average(mine)
        volume = volume/2 * 1000
        print("体重法，体积1: %.1f mL"%(volume))


        sizes = [3.8,14.7,14.7]
        sizes = np.array(sizes)
        print(np.prod(sizes))

        print(self.getVolume(1,1,1))
        # unit: cm
        r = 2.5

        # 13.5,15.6

        R = self.getRadius(8,4.5,5.1)
        print(R)
        l = 15.6

        v1 = self.getVolume2(13.5)
        v2 = self.getVolume2(10.4)
        print("量杯法，总体积: %.1f mL"%(v1+v2))

        return
class WechatDeal():
    """docstring for Deal"""
    def __init__(self):
        super(WechatDeal, self).__init__()

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

    def deal(self):
        """
        docstring for deal
        """
        # 从json文件中读取数据, data变量以字典形式
        # 存储数据
        filename = "好友列表.json"
        data = self.loadJson(filename)
        keys = ["nick_name", "remark_name",
                "user_name","wxid"]

        # 补全缺失字段
        # 有些数据缺少备注或者昵称
        # 用空格代替
        length = len(data)
        for i in range(length):
            for key in keys:
                if key not in data[i]:
                    data[i][key] = ""

        lines = ["昵称","备注","微信号","wxid"]

        workbook = xlwt.Workbook(encoding='utf-8')
        # 增加一个表格，命名为 好友列表
        booksheet = workbook.add_sheet("好友列表", cell_overwrite_ok=True)
        #写入第一行 "昵称","备注","微信号","wxid"
        for i in range(4):
            booksheet.write(0,i, lines[i])

        # 写入所有数据
        for i,item in enumerate(data):
            for j in range(4): 
                value = item[keys[j]]
                booksheet.write(i+1, j, value)

        # 输出到文件中
        filename = "灵江-好友列表.xls"
        print("输出表格: ",filename)
        workbook.save(filename)

        return
class Wechat():
    """
    class for wechat"""
    def __init__(self):
        super(Wechat, self).__init__()
    
    def write_txt_file(self,path, txt):
        '''
        写入txt文本
        '''

        with open(path, 'a', encoding='gb18030', newline='') as f:
            f.write(txt)

        return

    def read_txt_file(self,path):
        '''
        读取txt文本
        '''
        with open(path, 'r', encoding='gb18030', newline='') as f:
            return f.read()

        return

    def login(self):
        # 初始化机器人，扫码登陆
        bot = Bot()

        # 获取所有好友
        my_friends = bot.friends()

        print(type(my_friends))
        return my_friends

    def show_sex_ratio(self,friends):
        # 使用一个字典统计好友男性和女性的数量
        sex_dict = {'male': 0, 'female': 0}

        for friend in friends:
            # 统计性别
            if friend.sex == 1:
                sex_dict['male'] += 1
            elif friend.sex == 2:
                sex_dict['female'] += 1

        print(sex_dict)

    def show_area_distribution(self,friends):
        # 使用一个字典统计各省好友数量
        province_dict = {'北京': 0, '上海': 0, '天津': 0, '重庆': 0,
            '河北': 0, '山西': 0, '吉林': 0, '辽宁': 0, '黑龙江': 0,
            '陕西': 0, '甘肃': 0, '青海': 0, '山东': 0, '福建': 0,
            '浙江': 0, '台湾': 0, '河南': 0, '湖北': 0, '湖南': 0,
            '江西': 0, '江苏': 0, '安徽': 0, '广东': 0, '海南': 0,
            '四川': 0, '贵州': 0, '云南': 0,
            '内蒙古': 0, '新疆': 0, '宁夏': 0, '广西': 0, '西藏': 0,
            '香港': 0, '澳门': 0}

        # 统计省份
        for friend in friends:
            if friend.province in province_dict.keys():
                province_dict[friend.province] += 1

        # 为了方便数据的呈现，生成JSON Array格式数据
        data = []
        for key, value in province_dict.items():
            data.append({'name': key, 'value': value})

        print(data)

        return

    def show_signature(self,friends):
        # 统计签名
        for friend in friends:
            # 对数据进行清洗，将标点符号等对词频统计造成影响的因素剔除
            pattern = re.compile(r'[一-龥]+')
            filterdata = re.findall(pattern, friend.signature)
            write_txt_file('signatures.txt', ''.join(filterdata))

        # 读取文件
        content = read_txt_file('signatures.txt')
        segment = jieba.lcut(content)
        words_df = pd.DataFrame({'segment':segment})

        # 读取stopwords
        stopwords = pd.read_csv("stopwords.txt",
                                index_col=False,
                                quoting=3,
                                sep=" ",
                                names=['stopword'],
                                encoding='utf-8')
        words_df = words_df[~words_df.segment.isin(stopwords.stopword)]
        print(words_df)

        words_stat = words_df.groupby(by=['segment'])['segment'].agg({"计数":numpy.size})
        words_stat = words_stat.reset_index().sort_values(by=["计数"],ascending=False)

        # 设置词云属性
        # color_mask = imread('background.jfif')
        wordcloud = WordCloud(font_path="simhei.ttf",   # 设置字体可以显示中文
                        background_color="white",       # 背景颜色
                        max_words=100,                  # 词云显示的最大词数
                        # mask=color_mask,                # 设置背景图片
                        max_font_size=100,              # 字体最大值
                        random_state=42,
                        width=1000, height=860, margin=2,# 设置图片默认的大小,但是如果使用背景图片的话,                                                   # 那么保存的图片大小将会按照其大小保存,margin为词语边缘距离
                        )

        # 生成词云, 可以用generate输入全部文本,也可以我们计算好词频后使用generate_from_frequencies函数
        word_frequence = {x[0]:x[1]for x in words_stat.head(100).values}
        print(word_frequence)
        word_frequence_dict = {}
        for key in word_frequence:
            word_frequence_dict[key] = word_frequence[key]

        wordcloud.generate_from_frequencies(word_frequence_dict)
        # 从背景图片生成颜色值  
        image_colors = ImageColorGenerator(color_mask) 
        # 重新上色
        wordcloud.recolor(color_func=image_colors)
        # 保存图片
        wordcloud.to_file('output.png')
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()

    def main(self):
        friends = self.login()
        self.show_sex_ratio(friends)
        self.show_area_distribution(friends)
        self.show_signature(friends)

    def controlMouse(self):
        """
        docstring for controlMouse
        """

        from pynput.mouse import Button,Controller
        mouse = Controller( )
        mouse.position = (400,600)
        mouse.click(Button.left,10)

        return  

    def controlKeyboard(self):
        """
        docstring for controlKeyboard
        """
        from pynput.keyboard import Key,Controller
        keyboard = Controller()
        with keyboard.pressed(Key.cmd):
            keyboard.pressed("v")
            keyboard.release("v")

        keyboard.pressed(Key.enter)
        keyboard.release(Key.enter)
        return
    def test(self):
        """
        docstring for test
        """
        print("controld the mouse and keyboard")
        self.controlMouse()
        # self.controlKeyboard()
        return


