#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-05-18 00:16:22
    @project      : transform vim snippets sublime ones
    @version      : 0.2
    @source file  : vim_to_sublime_snippet.py

============================
"""

import os

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

            
# vim = VimToSublime()
# vim.get_all_snippets()