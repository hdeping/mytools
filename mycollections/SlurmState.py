#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-18 20:55:28
    @project      : get the status of sinfo
    @version      : 1.0
    @source file  : SlurmState.py

============================
"""

import os 
import numpy as np

class SlurmState():
    """
    get the status of the GPU server 
    with the command squeue, which is one 
    of the tool in slurm
    """
    def __init__(self):
        """
        self.split_lines:
            separation line strings
        self.prefix:
            formats of the output
        self.total:
            number of the remained GPUs
        self.running_jobs:
            number of the used GPUs
        self.waiting_num:
            number of the waiting jobs
        """
        super(SlurmState, self).__init__()
        self.split_lines  = '             ---------------------------------------------'
        self.prefix       = "             |%14s |     %d     |       %d       |"
        self.total        = 0
        self.running_jobs = 0
        self.waiting_num  = 0
        self.formats = [["             %d Jobs are running",
                         "             No Jobs are running"], 
                        ["             %d GPUs are available",
                         "             No GPUs are available"],
                        ["             %d Jobs Are Waiting"
                         "             No Jobs Are Waiting"]]

    def printLines(self,num):
        """
        docstring for printLines
        print the separation lines
        """
        for i in range(num):
            print(self.split_lines)
            
        return      
    def getSqueue(self):
        """
        get the results by running the command 
        squeue
        """
        squeue = os.popen('squeue').read()
        # deal with the squeue before spliting
        squeue = squeue.replace(', ',',')
        # spliting
        squeue = squeue.split()
        # to numpy type
        squeue = np.array(squeue)
        # reshape
        squeue = np.reshape(squeue,(-1,8))
        nodelist = squeue[1:,-1]
        return nodelist

    def getAllNodes(self):
        """
        get all the names of the nodes 
        in the GPU server
        """
        nodes = []
        nodes.append('controlmaster')
        nodes.append('slave3gpu1')
        for i in range(2,9):
            nodes.append('slave2gpu%d'%(i))
        state = {}
        for node in nodes:
            state[node] = 0
        return state

    def printNodeInfo(self,node_state,nodelist):
        """
        print the information of a computation node
        """
        # update the node info
        
        self.updateNodeInfo()

        self.printLines(2)
        # running jobs
        self.printRunningInfo(self.formats[0],self.running_jobs)
        # remained GPUs
        self.printRunningInfo(self.formats[1],self.total)
        self.printLines(2)
        # waiting jobs
        self.printRunningInfo(self.formats[2],self.waiting_num)
        return
    def updateNodeInfo(self):
        """
        docstring for updateNodeInfo
        """
        for node in nodelist:
            if node[0] == '(':
                self.waiting_num += 1
                continue
            node_state[node] += 1
        # output head 
        print('             |%14s | used gpus | remained gpus |'%(' node list  '))
        self.printLines(1)
        
        for node in node_state:
            used_number = node_state[node]
            self.running_jobs += used_number
            remained_number = 2 - used_number
            if node == 'slave3gpu1':
                remained_number = 3 - used_number
            self.total += remained_number
            
            if remained_number:
                print((self.prefix + " Available")%(node,used_number,remained_number))
            else:
                print(self.prefix%(node,used_number,remained_number))

        return

    def printRunningInfo(self, formats, num):
        """
        docstring for printRunningInfo
        input: 
            formats,string array of two strings
            num, integer value
        return: None    
        """
        
        if num:
            print(formats[0]%(num))
        else:
            print(formats[1])
            
        return
    def run(self):
        """
        docstring for run
        
        """
        # get all nodes
        node_state = self.getAllNodes()
        # get nodelist
        nodelist = self.getSqueue()
        #print(nodelist)
        self.printNodeInfo(node_state,nodelist)

        return
