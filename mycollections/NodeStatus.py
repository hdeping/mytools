#!/Users/huangdeping/miniconda3/bin/python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-30 10:18:02
    @project      : get the status of all the nodes
    @version      : 1.0
    @source file  : NodeStatus.py

============================
"""
import os

class NodeStatus():
    """
    get the status of all nodes in a server
    """
    def __init__(self):
        """
        self.nodes:
            all the nodes in a server, usually named 
            as node1, node2 etc.
        """
        super(NodeStatus, self).__init__()
        self.nodes = None 

    def getNodes(self):
        """
        docstring for getNodes
        input: 
            None
        return:
            None, but self.nodes was changed
            to ["node1",...,"node14"]
        """
        self.nodes = []
        for i in range(1,15):
            node = "node%d"%(i)
            self.nodes.append(node)

        return
    def getNodesRun(self):
        """
        analyze of the output after running the command
        pbsnodes
        input: 
            None
        return:
            dicts, a dictionary with nodes names as keys
            such as {"node1":[]...}
        """
        # get results from the command pbsnodes
        results = os.popen("pbsnodes")
        results = results.read()
        results = results.split("\n")

        dicts   = {}
        for line in results:
            if line[:4] == "node":
                key        = line
                dicts[key] = []
            line = line.split("=")
            name = line[0].replace(" ","")
            if name == "jobs":
                assert (key in dicts)
                dicts[key] = line[1].split(",")

        return dicts
    def run(self):
        """
        docstring for run
        print the results out
        """
        self.getNodes()
        dicts = self.getNodesRun()
        print("node run remain")
        for node in self.nodes:
            runNum    = len(dicts[node])
            remainNum = 28 - runNum
            print("%s %3d %3d"%(node,runNum,remainNum))
            

        return 

node = NodeStatus()
node.run()