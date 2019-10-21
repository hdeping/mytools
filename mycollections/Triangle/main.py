#!/usr/local/bin/python3

import numpy as np
from triangle import Triangle
import json
import matplotlib.pyplot as plt

points = [[3.0,0],[0.0,4.0],[0.0,0.0]]
#points = [[4.0,0],[0.0,4.0],[0.0,0.0]]
points = np.array(points)

tri = Triangle(points)

print(tri.get_vertices())

info = tri.get_info()
info = json.dumps(info,indent = 4)
print(info)

print("circum radius",tri.get_circum_center())
print("ortho  radius",tri.get_ortho_center())
#tri.draw_circum_circle()
tri.draw_inscribe_circle()
