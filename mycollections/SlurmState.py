#!/home/ncl/hdp/anaconda3/bin/python3

import os 
import numpy as np

def getSqueue():
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
def getAllNodes():
    nodes = []
    nodes.append('controlmaster')
    nodes.append('slave3gpu1')
    for i in range(2,9):
        nodes.append('slave2gpu%d'%(i))
    state = {}
    for node in nodes:
        state[node] = 0
    return state
def printNodeInfo(node_state,nodelist):
    # update the node info
    waiting_num = 0
    for node in nodelist:
        if node[0] == '(':
            waiting_num += 1
            continue
        node_state[node] += 1
    # output head
    print('             |%14s | used gpus | remained gpus |'%(' node list  '))
    print('             ---------------------------------------------')
    total = 0
    running_jobs = 0
    for node in node_state:
        used_number = node_state[node]
        running_jobs += used_number
        remained_number = 2 - used_number
        if node == 'slave3gpu1':
            remained_number = 3 - used_number
        total += remained_number
        if remained_number:
            print('             |%14s |     %d     |       %d       | Available'%(node,used_number,remained_number))
        else:
            print('             |%14s |     %d     |       %d       |'%(node,used_number,remained_number))
    print('             ---------------------------------------------')
    print('             ---------------------------------------------')
    # running jobs
    if running_jobs:
        print('             ',running_jobs,'Jobs are running')
    else:
        print('             No Jobs are running')
    # remained GPUs
    if total:
        print('             ',total,'GPUs are available')
    else:
        print('             No GPUs are available')
    print('             ---------------------------------------------')
    print('             ---------------------------------------------')
    if waiting_num:
        print('             ',waiting_num,"Jobs Are Waiting")
    else:
        print("             No Jobs Are Waiting")

# get all nodes
node_state = getAllNodes()
# get nodelist
nodelist = getSqueue()
#print(nodelist)
printNodeInfo(node_state,nodelist)
