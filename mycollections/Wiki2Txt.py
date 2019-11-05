#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-23 16:15:12
    @project      : transform the reference format
    @version      : 1.0
    @source file  : 05_deal.py

============================
"""

import re
import numpy as np

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
