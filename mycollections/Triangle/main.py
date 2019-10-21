#!/usr/local/bin/python3

import numpy as np

# parameters
unit = np.pi / 180.0
A    = 22.5*unit
B    = 45*unit
c    = 1
b    = np.sin(B)*c/np.sin(A+B)
a    = np.sin(A)*c/np.sin(A+B)
print(b,a)
print(b*np.cos(A),a*np.cos(B))
