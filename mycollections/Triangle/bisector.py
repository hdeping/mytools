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
            res1.append(line2[i] - line1[i])
            res2.append(line2[i] + line1[i])
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

    # get the inscribe line of A
    def get_inscribe_line_A(self):
        line = self.get_bisector_line(self.A,self.B,self.C)
        return line['interior']

    # get the inscribe line of B
    def get_inscribe_line_B(self):
        line = self.get_inscribe_line(self.B,self.A,self.C)
        return line['interior']

    # get the inscribe line of C
    def get_inscribe_line_C(self):
        line =  self.get_inscribe_line(self.C,self.A,self.B)
        return line['interior']

    # get the inscribe line information of the
    # triangle
    def get_inscribe_lines(self):
        line_A = self.get_inscribe_line_A()
        line_B = self.get_inscribe_line_B()
        line_C = self.get_inscribe_line_C()

        lines = {}
        lines["A"] = line_A
        lines["B"] = line_B
        lines["C"] = line_C
        return lines

    # get the inscribe line equation format of A
    def get_inscribe_line_string_A(self):
        line = self.get_inscribe_line_A()
        return self.get_line_string(line)

    # get the inscribe line equation format of B
    def get_inscribe_line_string_B(self):
        line = self.get_inscribe_line_B()
        return self.get_line_string(line)

    # get the inscribe line equation format of C
    def get_inscribe_line_string_C(self):
        line = self.get_inscribe_line_C()
        return self.get_line_string(line)

    # get the inscribe line equation format information
    # of the triangle
    def get_inscribe_line_strings(self):
        line_A = self.get_inscribe_line_string_A()
        line_B = self.get_inscribe_line_string_B()
        line_C = self.get_inscribe_line_string_C()

        lines = {}
        lines["A"] = line_A
        lines["B"] = line_B
        lines["C"] = line_C
        return lines

    # get the center of the inscribed circle of the 
    # triangle
    def get_inscribe_center(self):
        lines = self.get_inscribe_lines()
        line_B = lines["line B"] 
        line_A = lines["line A"] 
        res = self.get_intersect_point(line_A,line_B)
        return res

