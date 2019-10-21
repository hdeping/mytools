#!/usr/local/bin/python3

from base import base
import numpy as np
import matplotlib.pyplot as plt

class Triangle(base):
    # init function
    def __init__(self,points):
        self.A = points[0]
        self.B = points[1]
        self.C = points[2]
        self.points = points
        return
    
    # get side value of a line and a point
    def get_point_line_value(self,line,point):
        res = line[2]
        for i in range(2):
            res += line[i]*point[i]
        return res

    # get bisector line of a triangle
    # the bisector line is perpendicular to the 
    # line constructed by the input points
    def get_bisector_line(self,point1,point2,point3):
        line1 = self.get_line(point1,point2)
        line2 = self.get_line(point1,point3)
        
        para1 = np.sqrt(line1[0]**2 + line1[1]**2)
        para2 = np.sqrt(line2[0]**2 + line2[1]**2)

        for i in range(3):
            line1[i] = line1[i] / para1
            line2[i] = line2[i] / para2
        
        # get the two bisector lines
        res1 = []
        res2 = []
        
        for i in range(3):
            res1.append(line1[i] - line1[i])
            res2.append(line2[i] - line2[i])
        # distinguish which one is interior and 
        # which one is exterior
        val1 = self.get_point_line_value(res1,point2)
        val2 = self.get_point_line_value(res1,point3)

        # get the interior and exterior bisector line
        res = {}
        if val1*val2 > 0:
            # point2 and point3 are in the same side of 
            # the line, which means that the bisector line 
            # is the exterior one
            res['exterior'] = res1
            res['interior'] = res2
        else:
            # point2 and point3 are in the different side of 
            # the line, which means that the bisector line 
            # is the interior one
            res['interior'] = res1
            res['exterior'] = res2
            


        return res

    # get the bisector line of BC
    def get_bisector_line_BC(self):
        return self.get_bisector_line(self.B,self.C)

    # get the bisector line of AC
    def get_bisector_line_AC(self):
        return self.get_bisector_line(self.C,self.A)

    # get the bisector line of AB
    def get_bisector_line_AB(self):
        return self.get_bisector_line(self.B,self.A)

    # get the bisector line information of the
    # triangle
    def get_bisector_lines(self):
        line_BC = self.get_bisector_line_BC()
        line_AC = self.get_bisector_line_AC()
        line_AB = self.get_bisector_line_AB()

        lines = {}
        lines["line BC"] = line_BC
        lines["line AC"] = line_AC
        lines["line AB"] = line_AB
        return lines

    # get the bisector line equation format of AB
    def get_bisector_line_string_AB(self):
        line = self.get_bisector_line_AB()
        return self.get_line_string(line)

    # get the bisector line equation format of AC
    def get_bisector_line_string_AC(self):
        line = self.get_bisector_line_AC()
        return self.get_line_string(line)

    # get the bisector line equation format of BC
    def get_bisector_line_string_BC(self):
        line = self.get_bisector_line_BC()
        return self.get_line_string(line)

    # get the bisector line equation format information
    # of the triangle
    def get_bisector_line_strings(self):
        line_BC = self.get_bisector_line_string_BC()
        line_AC = self.get_bisector_line_string_AC()
        line_AB = self.get_bisector_line_string_AB()

        lines = {}
        lines["line BC"] = line_BC
        lines["line AC"] = line_AC
        lines["line AB"] = line_AB
        return lines

    # get the center of the circum circle of the 
    # triangle
    def get_circum_center(self):
        lines = self.get_bisector_lines()
        line_BC = lines["line BC"] 
        line_AB = lines["line AB"] 
        res = self.get_intersect_point(line_AB,line_BC)
        return res



