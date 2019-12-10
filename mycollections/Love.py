#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : xiaohengye
    @mail address : xiaohengye@blackhole.com
    @date         : 10019-12-10
    @project      : love record of the blackhole
    @version      : 1.0
    @source file  : main.py

============================
"""

import hashlib
import numpy as np
import sys

class Love():

    """Docstring for Love. """

    def __init__(self):
        """
        self.string0:
            initial text information
        self.number :
            size of search space
        self.target :
            "5201314"
        self.length :
            length of the target number string
        """
        self.string0 = "亲爱的小阮，我愿意爱你爱到宇宙的尽头。小黄。"
        self.number  = 1 << 30
        self.target  = "5201314"
        self.length  = len(self.target)
        return
    def getHash(self, string):
        """TODO: Docstring for getHash.

        :string: TODO
        :returns: TODO

        """
        string = "%s5201314%s"%(self.string0,string)
        code   = string.encode()
        out = hashlib.sha256(code).hexdigest()
        return out

    def run(self, cycles):
        """TODO: Docstring for run.

        :cycles: 
            0,1,and so on, 
            range: self.number * cycles,self.number*(cycles + 1)
        :returns: TODO

        """
        num1 = self.number * cycles
        num2 = num1 + self.number
        for i in range(num1,num2):
            array  = np.random.randint(0,10,11)
            array  = array.astype("str")
            array  = "".join(array)
            out    = self.getHash(array)
            if out[:self.length] == self.target:
                print("find you: ",i,string,out)
                break
            if i%(1<<22) == 0:
                print("section ",i,string,out)
            #print(out[:7])
                
        return
    def test(self):
        """TODO: Docstring for test.
        :returns: TODO

        """
        strings = ["56572066746","30083723179"]
        
        for string in strings:
            print(self.target,string)
            print(self.getHash(string))
        return

love = Love()
#love.run(0)
love.test()

