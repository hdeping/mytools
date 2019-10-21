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
    
    # get vertical line of two points 
    # the vertical line is perpendicular to the 
    # line constructed by the input points
    def get_vertical_line(self,point1,point2):
        mid_point = self.get_mid_point(point1,point2)
        line      = self.get_line(point1,point2)
        res       = self.get_ortho_line(line,mid_point)
        return res

    # get the vertical line of BC
    def get_vertical_line_BC(self):
        return self.get_vertical_line(self.B,self.C)

    # get the vertical line of AC
    def get_vertical_line_AC(self):
        return self.get_vertical_line(self.C,self.A)

    # get the vertical line of AB
    def get_vertical_line_AB(self):
        return self.get_vertical_line(self.B,self.A)

    # get the vertical line information of the
    # triangle
    def get_vertical_lines(self):
        line_BC = self.get_vertical_line_BC()
        line_AC = self.get_vertical_line_AC()
        line_AB = self.get_vertical_line_AB()

        lines = {}
        lines["line BC"] = line_BC
        lines["line AC"] = line_AC
        lines["line AB"] = line_AB
        return lines

    # get the vertical line equation format of AB
    def get_vertical_line_string_AB(self):
        line = self.get_vertical_line_AB()
        return self.get_line_string(line)

    # get the vertical line equation format of AC
    def get_vertical_line_string_AC(self):
        line = self.get_vertical_line_AC()
        return self.get_line_string(line)

    # get the vertical line equation format of BC
    def get_vertical_line_string_BC(self):
        line = self.get_vertical_line_BC()
        return self.get_line_string(line)

    # get the vertical line equation format information
    # of the triangle
    def get_vertical_line_strings(self):
        line_BC = self.get_vertical_line_string_BC()
        line_AC = self.get_vertical_line_string_AC()
        line_AB = self.get_vertical_line_string_AB()

        lines = {}
        lines["line BC"] = line_BC
        lines["line AC"] = line_AC
        lines["line AB"] = line_AB
        return lines

    # get the center of the circum circle of the 
    # triangle
    def get_circum_center(self):
        lines = self.get_vertical_lines()
        line_BC = lines["line BC"] 
        line_AB = lines["line AB"] 
        res = self.get_intersect_point(line_AB,line_BC)
        return res



