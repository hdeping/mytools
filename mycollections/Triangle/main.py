#!/usr/local/bin/python3

import numpy as np
from triangle import Triangle
import json

points = [[3.0,0],[0.0,4.0],[0.0,0.0]]
#points = [[3.0,0],[2.0,0.0],[1.0,0.0]]
points = np.array(points)
print(points)


tri = Triangle(points)
print(tri.A)
print(tri.B)
print(tri.C)

sideLengths = tri.get_laterals()
print(sideLengths)
angles = tri.get_angles()
print(angles)
lines = tri.get_lines()
lines = json.dumps(lines,indent = 4)
print(lines)
lines = tri.get_line_strings()
lines = json.dumps(lines,indent = 4)
print(lines)
tri.draw()
