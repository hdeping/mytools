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
    
    # get the inscribe line of A
    def get_inscribe_line_A(self):
        line = self.get_bisector_line(self.A,self.B,self.C)
        return line['interior']

    # get the inscribe line of B
    def get_inscribe_line_B(self):
        line = self.get_bisector_line(self.B,self.A,self.C)
        return line['interior']

    # get the inscribe line of C
    def get_inscribe_line_C(self):
        line =  self.get_bisector_line(self.C,self.A,self.B)
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
        line_B = lines["B"] 
        line_A = lines["A"] 
        res = self.get_intersect_point(line_A,line_B)
        return res

