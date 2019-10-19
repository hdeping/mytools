#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-19 16:43:55
    @project      : transfer .ris file to a .bib one
    @version      : 1.0
    @source file  : Ris2Bib.py

============================
"""


import sys

class Ris2Bib():
    """
    transfer .ris file to a .bib one
    """
    def __init__(self):
        """
        self.risfile:
            name of the .ris file
        self.author_list:
            lists for the authors
        self.title:
            title of the reference
        self.journal:
            journal name 
        self.volume:
            volume of the reference
        self.year:
            published year
        self.month:
            published month
        self.startingpage:
            starting page of the reference
        self.finalpage:
            ending page of the reference
        self.publisher:
            publisher of the document
        self.doi:
            doi of the document
        self.abstract:
            abstract of the document
        self.url:
            url of the document
        """
        super(Ris2Bib, self).__init__()
        self.getRisfile()
            
        self.author_list   = []
        self.title         = None
        self.journal       = None
        self.volume        = None
        self.year          = None
        self.month         = None
        self.startingpage  = None
        self.finalpage     = None
        self.publisher     = None
        self.doi           = None
        self.abstract      = None
        self.url           = None
        self.ris2Bib = {
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
            "UR" : "url"
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
                    value = data[1].strip(' ').strip('\n').strip('\r')
                    #print case
                    if field == 'AU':
                        self.author_list.append(value)
                    elif field == 'TI':
                        self.title = value
                    elif field == 'JA' or field == 'JO':
                        self.journal = value
                    elif field == 'VL':
                        self.volume = value
                    elif field == 'PY':
                        self.year=value.rsplit('/')[0]
                        #month=value.rsplit('/')[1]
                    elif field == 'SP':
                        self.startingpage = value
                    elif field == 'EP':
                        self.finalpage = value
                    elif field == 'L3' or field == 'DO':
                        self.doi = value
                    elif field == 'PB':
                        self.publisher = value
                    elif field == 'AB':
                        self.abstract = value
                    elif field == 'UR':
                        self.url = value
        return
    def tranDocInfo(self):
        """
        docstring for tranDocInfo
        return: 
            lines, string array, contents of  the bib file
        """
        # dealing with the data
        lines=[]
        firstauthor = self.author_list[0].rsplit(',')[0].strip(' ')
        name = (firstauthor.lower(),self.year)
        lines.append('@article{%s%s,' % name)
        authors   = ' and '.join(self.author_list)
        authorline = "    author = {%s}," % authors 
        lines.append(authorline)
        if self.title is not None:
            lines.append("    title = {%s}," % self.title)
        if self.journal is not None:
            lines.append("    journal = {%s}," % self.journal)
        if self.volume is not None:
            lines.append("    volume = {%s}," % self.volume)
        if self.startingpage is not None and self.finalpage is not None:
            pages = (self.startingpage,self.finalpage)
            lines.append("    pages = {%s--%s}," % pages)
        if self.year is not None:
            lines.append("    year = {%s}," % self.year)
        if self.doi is not None:
            lines.append("    doi = {%s}," % self.doi)
        # publisher
        if self.publisher is not None:
            lines.append("    publisher = {%s}," % self.publisher)
        # abstract
        if self.abstract is not None:
            lines.append("    abstract = {%s}," % self.abstract)
        # url
        if self.url is not None:
            lines.append("    url = {%s}," % self.url)
            
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

ris = Ris2Bib()
ris.run()