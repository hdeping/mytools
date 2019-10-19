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
    def __init__(self,filename):
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

        self.risfile       = filename
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
        firstauthor=author_list[0].rsplit(',')[0].strip(' ')

        lines.append('@article{'+firstauthor.lower()+year+',')

        authorline=' '*4 + 'author={' + ' and '.join(author_list)+'},'
        lines.append(authorline)
        if title is not None:
            lines.append(' '*4 + 'title={' + title + '},')
        if journal is not None:
            lines.append(' '*4 + 'journal={' + journal + '},')
        if volume is not None:
            lines.append(' '*4 + 'volume={' + volume + '},')
        if startingpage is not None and finalpage is not None:
            lines.append(' '*4 + 'pages={' + startingpage + '--'+finalpage+'},')
        if year is not None:
            lines.append(' '*4 + 'year={' + year + '},')
        if doi is not None:
            lines.append(' '*4 + 'doi={' + doi + '},')
        # publisher
        if publisher is not None:
            lines.append(' '*4 + 'publisher={' + publisher + '},')
        # abstract
        if abstract is not None:
            lines.append(' '*4 + 'abstract={' + abstract + '},')
        # url
        if url is not None:
            lines.append(' '*4 + 'url={' + url + '}')
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

ris = Ris2Bib(sys.argv[1])
ris.run()