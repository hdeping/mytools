#!/usr/local/bin/python3

import numpy as np
from triangle import Triangle
import json
import matplotlib.pyplot as plt

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
#tri.draw()
print(tri.get_area())
radiuses = tri.get_radiuses()
radiuses = json.dumps(radiuses,indent = 4)
print(radiuses)
#print(tri.get_escribe_radius_A())
#print(tri.get_escribe_radius_B())
#print(tri.get_escribe_radius_C())

#tri.draw()

c = tri.get_mid_point_AB()
print(tri.get_mid_points())
print(tri.get_gravity_center())
print(tri.get_mid_lines())
print(tri.get_mid_line_strings())

line1 = [1,1,1]
line2 = [1,2,2]
print(tri.get_intersect_point(line1,line2))
tri.test_get_intersect_point()
#tri.draw_ortho_lines()
#print(tri.get_ortho_points())
#tri.draw_ortho_lines()
#tri.draw_circum_circle()
tri.draw_escribe_circle()

