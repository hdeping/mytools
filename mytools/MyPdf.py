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
from PyPDF2 import PdfFileReader
from PyPDF2 import PdfFileWriter
from PyPDF2 import PdfFileMerger

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

    def mergePdfs(self,filenames,mergedName,passwords=None):
        """
        docstring for mergePdfs
        filenames:
            1d array of filenames
        mergedName:
            name of the output file
        passwords:
            passwords for the pdf files if is needed
        """
        # number of the files
        num = len(filenames)
        # note that False should be used
        pdf_merger = PdfFileMerger()
    
        for i in range(num):
            print("adding ",filenames[i])
            fp =  open(filenames[i],"rb") 
            pdf_reader = PdfFileReader(fp,strict=False)
            if not pdf_reader:
                return
            pdf_merger.append(pdf_reader)
            fp.close()

        with open(mergedName, 'wb') as fp:
            print("output to ",mergedName)
            pdf_merger.write(fp)

        return
    def pick100(self,threshold=100):
        """
        docstring for getPages
        get all the pages of the input pdf files
        with over 100 pages from the command line
        """  
        import os 

        names = os.listdir()
        num = len(names)
        pages = 0
        count = 0
        for j in range(1,num):
            try:
                input = PdfFileReader(open(names[j],"rb"))
                i = input.numPages
                if i > threshold:
                    print("%s, %d pages"%(names[j],i))
                    count += 1
            except Exception:
                print("something is wrong with",names[j])
        print("there are %d books over %d pages"%(count,threshold))
        return
