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
    def test(self):
        """
        test all the classes in the mytools module
        """
        self.testDrawPig()
        self.testMyGUI()

test = Test()
test.test()

        