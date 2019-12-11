#!/usr/bin/env python
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-12-11 15:49:24
    @project      : wrapper for bitcoin-cli
    @version      : 1.0
    @source file  : Bitcoin.py

============================
"""

import json
import os
from mytools import MyCommon
import time

class Bitcoin(MyCommon):
    """
    wrapper for bitcoin-cli"""
    def __init__(self):
        """
        self.command:
            command for bitcoin-cli
        """
        super(Bitcoin, self).__init__()
        self.command = "bitcoin-cli getblock `bitcoin-cli getblockhash %d`"
        self.keys    = ["size","weight","height",
                        "time","nonce","ntx","difficulty"]
        return

    def getInfo(self,content):
        dicts = {}
        for key in self.keys:
            dicts[key] = content[key]

        return dicts

    def getDate(self,timestamp):
        """
        timestamp: 
            such as 1354216278
        return:
            date 
        """
        timeArray = time.localtime(timestamp)
        date = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)

        return date 
    def run(self):
        for i in range(0,1,100000):
            command = self.command % (i)
            content = os.popen(command).read()
            content = json.loads(content)

            print(i,content)


        return


bitcoin = Bitcoin()
bitcoin.run()

        