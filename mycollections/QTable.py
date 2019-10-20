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

# initialize the state array
state = np.zeros(16)
state[6] = -1
state[9] = -1
state[10] = 1

# not all the positive share the same kinds of operations
# some positions only have two ones, some others have three

# total operations: 4*16 - 4*4 = 48
# 48 action-value   Q values
#
# 4*2 + 8*3 + 4*4 = 48
QTable = np.zeros((16,4))

#print(QTable)
#print("state")
#print(state)
# impossible operation
"""
0 1 2 3 no up
0 4 8 12 no left
3 7 11 15 no right
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
for epoch in range(10):
    ii = 0
    while not (ii == 6 or ii == 9 or ii == 10):
        print(ii)
        ii = ii + 1








