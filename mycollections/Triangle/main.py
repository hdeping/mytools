#!/usr/local/bin/python3

import numpy as np
from triangle import Triangle

points = [[3.0,0],[0.0,4.0],[0.0,0.0]]
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
print(tri.get_lines())
print(tri.get_line_strings())
