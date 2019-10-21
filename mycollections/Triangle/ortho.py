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
    
    # get ortho line of a point and a line
    # the ortho line is perpendicular to the input line
    def get_ortho_line(self,line,point):
        res = []
        res.append(line[1])
        res.append(- line[0])
        c = - (res[0]*point[0] + res[1]*point[1])
        return res

    # get the ortho line of A-BC
    def get_ortho_line_A_BC(self):
        line = self.get_line_BC()
        return self.get_line(line,self.A)

    # get the ortho line of B-AC
    def get_ortho_line_B_AC(self):
        line = self.get_line_AC()
        return self.get_line(line,self.B)

    # get the ortho line of C-AB
    def get_ortho_line_C_AB(self):
        line = self.get_line_AB()
        return self.get_line(line,self.C)

    # get the ortho line information of the
    # triangle
    def get_ortho_lines(self):
        line_A_BC = self.get_ortho_line_A_BC()
        line_B_AC = self.get_ortho_line_B_AC()
        line_C_AB = self.get_ortho_line_C_AB()

        lines = {}
        lines["line A-BC"] = line_A_BC
        lines["line B-AC"] = line_B_AC
        lines["line C-AB"] = line_C_AB
        return lines

    # get the ortho line equation format of C_AB
    def get_ortho_line_string_C_AB(self):
        line = self.get_ortho_line_C_AB()
        return self.get_line_string(line)

    # get the ortho line equation format of B-AC
    def get_ortho_line_string_B_AC(self):
        line = self.get_ortho_line_B_AC()
        return self.get_line_string(line)

    # get the ortho line equation format of A-BC
    def get_ortho_line_string_A_BC(self):
        line = self.get_ortho_line_A_BC()
        return self.get_line_string(line)

    # get the ortho line equation format information
    # of the triangle
    def get_ortho_line_strings(self):
        line_A_BC = self.get_ortho_line_string_A_BC()
        line_B_AC = self.get_ortho_line_string_B_AC()
        line_C_AB = self.get_ortho_line_string_C_AB()

        lines = {}
        lines["line A-BC"] = line_A_BC
        lines["line B-AC"] = line_B_AC
        lines["line C-AB"] = line_C_AB
        return lines

    # get ortho point of a point and a line
    def get_ortho_point(self,line,point):
        ortho_line = self.get_ortho_line(line,point)
        res = self.get_intersect_point(line,ortho_line)
        return res

    # get ortho point of AB
    def get_ortho_point_C_AB(self):
        line = self.get_line_AB()
        return self.get_ortho_point(line,self.C)

    # get ortho point of AC
    def get_ortho_point_B_AC(self):
        return self.get_ortho_point(self.A,self.C)

    # get ortho point of BC
    def get_ortho_point_A_BC(self):
        return self.get_ortho_point(self.C,self.B)

    # ortho point information of the triangle
    def get_ortho_points(self):
        point_C_AB = self.get_ortho_point_C_AB()
        point_B_AC = self.get_ortho_point_B_AC()
        point_A_BC = self.get_ortho_point_A_BC()
        points = {}
        points["point C-AB"] = point_C_AB
        points["point B-AC"] = point_B_AC
        points["point A-BC"] = point_A_BC
        return points
