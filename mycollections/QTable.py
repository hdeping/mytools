# there are 16 states

"""
0  1  2  3
4  5  6  7
8  9  10 11
12 13 14 15

state = 0
state6 = state9 = - 1
state10 = 1

"""

import numpy as np

# initialize the value array
value = np.zeros(16)
value[6] = -1
value[9] = -1
value[10] = 1

# not all the positive share the same kinds of operations
# some positions only have two ones, some others have three

# total operations: 4*16 - 4*4 = 48
# 48 action-value   Q values
#
# 4*2 + 8*3 + 4*4 = 48
QTable = np.zeros((16,4))
# updating states
# 0,1 left,right -/+ 1
# 2,4 up,down    -/+ 4

#print(QTable)
#print("value")
#print(value)
# impossible operation

"""
0  1  2  3 no up
0  4  8  12 no left
3  7  11 15 no right
12 13 14 15 no down
"""

import numpy as np
import  time
#  get random seed
np.random.seed(int(time.time()))

# gamma : decay factor (0,1)
# epsilon:  (0,1) probability, best or random
# lr: learning rate
# q(s,a) = q(s,a) + lr*(r_{t+1}+gamma*max q(s_t{t+1},a') - q(s,a))
def actionNumber():
    actions = np.ones(16)*3
    corner = [0,3,12,15]
    center = [5,6,9,10]
    # corner: 2
    for i in corner:
        actions[i] = 2

    # center: 3
    for i in center:
        actions[i] = 4

    return actions

actions = actionNumber()

def getNewState(input_state):

    ii = 0
    state  = input_state
    if ii == 0:
        state = state - 1
    elif ii == 1:
        state = state + 1
    elif ii == 2:
        state = state - 4
    elif ii == 3:
        state = state + 4
    return state


for epoch in range(10):
    state = 0
    while not (state == 6 or state== 9 or state == 10):
        print(state)
        state = getNewState(state)


