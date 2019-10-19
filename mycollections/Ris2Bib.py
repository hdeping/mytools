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

ris = Ris2Bib()
ris.run()