#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-06-09 13:54:18
    @project      : xml to json
    @version      : 1.0
    @source file  : read.py

============================
"""
from xml.etree.ElementTree import parse
from xml.etree.ElementTree import ElementTree

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

tree = AnalyzeTree()
tree.test()
