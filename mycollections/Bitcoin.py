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
import numpy as np
import matplotlib.pyplot as plt


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
                        "time","nonce","nTx","difficulty"]
        # self.blocksNum = 500
        self.blocksNum = 607625
        return

    def getInfo(self,content):
        dicts = {}
        for key in self.keys:
            dicts[key] = content[key]

        dicts["date"] = self.getDate(content["time"])

        return dicts

    def getDate(self,timestamp):
        """
        timestamp: 
            such as 1354216278
        return:
            date 
        """
        timeArray = time.localtime(timestamp)
        date = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)

        return date 
    def run(self):
        results = {}
        t1 = time.time()
        for i in range(self.blocksNum):
            command = self.command % (i)
            content = os.popen(command).read()
            content = json.loads(content)

            key     = content["hash"]
            results[key] = self.getInfo(content)
            if i % 100000 == 0:
                print(i)
                self.writeJson(results,"bitcoin%d.json"%(i))

                
        self.writeJson(results,"bitcoin.json")
        print(time.time() - t1)

        return
    def test(self):
        filename = "bitcoin.json"
        data = self.loadJson(filename)
        # print(data)
        y = []
        z = []
        blockTime = []
        for key in data:
            y.append(data[key]["nTx"])
            z.append(data[key]["difficulty"])
            blockTime.append(data[key]["time"])
        # print(y)
        freq = 1000
        num = (len(y) // freq)*freq
        y = np.array(y[:num])
        y = y.reshape((-1,freq))
        y = np.average(y,axis=1)
        x = np.arange(len(y))*10*freq/60/24/365
        
        z = np.array(z[:num])
        z = z.reshape((-1,freq))
        z = np.average(z,axis=1)
        # z = np.log10(z)
        # y = np.log10(y)

        plt.semilogy(x,y,lw = self.width)
        plt.semilogy(x,z,lw = self.width)
        plt.show()
        blockTime = np.array(blockTime)
        blockTime = blockTime / 60
        print(blockTime)
        blockTime = blockTime[1:] - blockTime[:-1]
        
        #plt.hist(blockTime)
        # plt.show()
        print(blockTime)
        print(min(blockTime))
        print(max(blockTime))
        for i,item in enumerate(blockTime):
            if item < 0:
                print(i,item)
                break
        print("average time",np.average(blockTime))
        sections = np.arange(20)
        cuts = pd.cut(blockTime,sections)
        counts = pd.value_counts(cuts)
        print(counts)
        counts.plot.bar()
        # plt.show()


        return


bitcoin = Bitcoin()
# bitcoin.run()
bitcoin.test()

        