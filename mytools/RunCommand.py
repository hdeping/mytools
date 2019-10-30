#!/Users/huangdeping/miniconda3/bin/python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-24 13:54:28
    @project      : wrapper for the shell command
    @version      : 0.1
    @source file  : RunCommand.py

============================
"""
import os
import sys

class RunCommand():
    """
    run the shell command with os.system
    """
    def __init__(self):
        super(RunCommand, self).__init__()
    def gitRebase(self,num):
        """
        docstring for gitRebase
        input: 
            num, an integer number, last commit numbers
        return:
            None, but the command git rebase -i HEAD~num
            was executed
        """
        command = "git rebase -i HEAD~%d"%(num)
        print(command)
        os.system(command)
        return
    def runGitRebase(self):
        """
        docstring for runGitRebase
        run self.gitRebase accepted 
        a argument from the command line
        """
        try:
            num = int(sys.argv[1])
            self.gitRebase(num)
        except IndexError:
            print("you need a command line argument")
        return
        