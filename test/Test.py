#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-09 16:23:52
    @project      : test for mytools module
    @version      : 1.0
    @source file  : Test.py

============================
"""

import mytools
import numpy as np

class Test(object):
    """
    It is a class for testing the module mytools
    """
    def __init__(self):
        super(Test, self).__init__()
    def testDrawPig(self):
        """
        It is used to test DrawPig class
        """
        test = mytools.DrawPig()
        test.cutePig()

        return

    def testMyGUI(self):
        """
        docstring for testMyGUI
        It is used to test MyGUI class
        """
        test = mytools.MyGUI()
        test.interface()

        return
    def testMyCommon(self):
        """
        docstring for testMyCommon
        It is used to test the MyCommon class
        """
        test = mytools.MyCommon()
        filename = "test.json"
        data = self.getTestData()
        test.writeFile(data,filename)
        filename = filename.replace("json","yml")
        test.writeFile(data,filename)

        data = test.loadFile(filename)
        print(data)

        filename = "data.txt"
        data     = test.loadStrings(filename)
        result   = test.getStringStati(data)
        result   = test.sortDicts(result)
        print(result)



        return
    def getTestData(self):
        """
        docstring for getTestData
        return: a dictionary with random string as 
                keys,such as 
                {"JCXJICNICK":[74,67,88,74,73,67,78,73,67,75]...}
        """
        data = {}
        rand = np.random.randint(65,91,100)
        rand = rand.reshape((10,10))

        for value in rand:
            key = self.numbers2string(value)
            data[key] = value.tolist()

        return data

    def numbers2string(self,array):
        """
        docstring for numbers2string
        [65,65,66] --> "AAB"
        """
        results = []
        for i in array:
            results.append(chr(i))
        results = "".join(results)
        return results
    def test(self):
        """
        test all the classes in the mytools module
        """
        # self.testDrawPig()
        # self.testMyGUI()
        self.testMyCommon()

test = Test()
test.test()

        
