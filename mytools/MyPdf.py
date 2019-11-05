#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-11-05 10:32:12
    @project      : my own pdf module
    @version      : 1.0
    @source file  : MyPdf.py

============================
"""

import sys
from PyPDF2 import PdfFileReader,PdfFileWriter

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
