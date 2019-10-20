#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-20 16:24:38
    @project      : test for Q-table
    @version      : 1.0
    @source file  : QTable.py

============================
"""
import numpy as np 
import time


class QTable():
    """
    there are 16 states
    0  1  2  3
    4  5  6  7
    8  9  10 11
    12 13 14 15

    reward = 0
    reward6 = reward9 = - 1
    reward10 = 1

    """
    def __init__(self):
        """
        """
        super(QTable, self).__init__()
        #  get random seed
        np.random.seed(int(time.time()))

        # initialize the value array
        self.reward = np.zeros(16)
        self.reward[6] = -1
        self.reward[9] = -1
        self.reward[10] = 1
        self.reward[14] = -1
        self.actions = ['left', 'right', 'down', 'up']
        # hyper parameters
        self.epsilon = 0.9
        self.gamma   = 0.9
        self.lr      = 0.1

        return



    def initQValue():
        """
        not all the positive share the same kinds of operations
        some positions only have two ones, some others have three

        total operations: 4*16 - 4*4 = 48
        48 action-value   Q values
        
        4*2 + 8*3 + 4*4 = 48
        QTable = np.zeros((16,4))
        updating states
        0,1 left,right -/+ 1
        2,4 up,down    -/+ 4

        print(QTable)
        print("value")
        print(value)
        impossible operation
        """
        # (ii,jj) 4*ii + jj
        table = []
        # four actions of all states
        for i in range(16):
            dictionary = {}
            for j in range(4):
                dictionary[actions[j]] = 0
            table.append(dictionary)

        for i in range(4):
            # fist row
            state = i
            table[state].pop('up')
            # fourth row
            state = 12 + i
            table[state].pop('down')
            # fist column
            state = i * 4
            table[state].pop('left')
            # fourth column
            state = i * 4 + 3
            table[state].pop('right')

        return table


    def table_print(table):
        """
        print the Q-table
        """
        string = ""
        for i, line in enumerate(table):
            #print(i, line.values())

            string = "%d,"%(i)
            values = [value for value in line.values()]
            #string = "%s,%.3f,"%(string,sum(values))
            #print("%6.3f"%(sum(values)))
            #if i%4 == 3:
            #    print(string)
            #    string  = ""
            for value in line.values():
                string = "%s,%.3f,"%(string,value)
            print(string)
        return
    def updateStates(input_state, action):
        """
        0  1  2  3  no up
        0  4  8  12 no left
        3  7  11 15 no right
        12 13 14 15 no down
        gamma : decay factor (0,1)
        epsilon:  (0,1) probability, best or random
        lr: learning rate
        q(s,a) = q(s,a) + lr*(r_{t+1}+gamma*max q(s_{t+1},a') - q(s,a))
        """
        if action == 'left':
            state1 = input_state - 1
        elif action == 'right':
            state1 = input_state + 1
        elif action == 'down':
            state1 = input_state + 4
        elif action == 'up':
            state1 = input_state - 4

        assert state1 >= 0 and state1 <= 15

        #print("updateStates",action,state1,"input state",input_state)
        return state1


    # update states
    def updateValues(self,input_state):
        q_table = QTable[input_state]
        for action in q_table:
            #print("action",action,input_state)
            new_state = updateStates(input_state, action)
            line = QTable[new_state]
            # action list
            items = line.items()
            # value list
            actions = [key for key, value in items]
            values = [value for key, value in items]
            #values = np.array(values)

            # Bellmann equation
            q_table[action] += lr * (reward[new_state] + gamma * max(values) - q_table[action])


    def getNewState(self,input_state):
        #
        #  update the values
        updateValues(input_state)
        line = QTable[input_state]
        items = line.items()
        # value list
        actions = [key for key, value in items]
        values =  [value for key, value in items]
        values = np.array(values)

        if np.random.rand() < epsilon:
            # get the best action
            action_index = np.argmax(values)
        else:
            # get the random action
            action_index = np.random.randint(len(line))

        # get the action

        action = actions[action_index]

        state = updateStates(input_state, action)

        return state

    def run(self):
        """
        docstring for run

        """
        qTable = self.initQValue()
        return

    result = []
    cycles = 10000
    for epoch in range(cycles):
        #print("epoch", epoch)
        state = 0
        while not (state == 6 or state == 9 or state == 10 or state == 14):

            state = getNewState(state)

            #print("state", state)
        if epoch > 10000*0.8:
            epsilon = 1.0
        #print(state)
        result.append(state)
        #table_print(QTable)
    result = np.array(result)
    result = np.reshape(result,(100,100))

    filename = "data.txt"
    output = np.zeros((100,2))
    for i,arr in enumerate(result):
        ii = sum(arr==10)
        print(i,ii)
        output[i,:] = [i,ii]
        
    np.savetxt(filename,output,fmt="%d,%d")
    table_print(QTable)


