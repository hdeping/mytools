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
    
    # get mid point of two points
    def get_mid_point(self,coor1,coor2):
        x = (coor1[0] + coor2[0])/2
        y = (coor1[1] + coor2[1])/2
        res = [x,y]
        return res

    # get mid point of AB
    def get_mid_point_AB(self):
        return self.get_mid_point(self.A,self.B)

    # get mid point of AC
    def get_mid_point_AC(self):
        return self.get_mid_point(self.A,self.C)

    # get mid point of BC
    def get_mid_point_BC(self):
        return self.get_mid_point(self.C,self.B)

    # mid point information of the triangle
    def get_mid_points(self):
        point_AB = self.get_mid_point_AB()
        point_AC = self.get_mid_point_AC()
        point_BC = self.get_mid_point_BC()
        points = {}
        points["AB"] = point_AB
        points["AC"] = point_AC
        points["BC"] = point_BC
        return points

    # get the gravity center
    def get_gravity_center(self):
        center = np.mean(self.points,axis = 0)
        return center

    # draw the mid lines
    def draw_mid_lines(self):
        plt.figure(figsize = (9,9))
        plt.axis("equal")

        self.draw_line(self.A,self.B)
        self.draw_line(self.A,self.C)
        self.draw_line(self.C,self.B)

        points   = self.get_mid_points()
        point_AB =  points["AB"]
        point_AC =  points["AC"]
        point_BC =  points["BC"]

        self.draw_line(self.A,point_BC)
        self.draw_line(self.B,point_AC)
        self.draw_line(self.C,point_AB)

        plt.show()

    # get the mid line of A-BC
    def get_mid_line_A_BC(self):
        point_BC = self.get_mid_point_BC()
        return self.get_line(self.A,point_BC)

    # get the mid line of B-AC
    def get_mid_line_B_AC(self):
        point_AC = self.get_mid_point_AC()
        return self.get_line(self.B,point_AC)

    # get the mid line of C-AB
    def get_mid_line_C_AB(self):
        point_AB = self.get_mid_point_AB()
        return self.get_line(self.C,point_AB)

    # get the mid line information of the
    # triangle
    def get_mid_lines(self):
        line_A_BC = self.get_mid_line_A_BC()
        line_B_AC = self.get_mid_line_B_AC()
        line_C_AB = self.get_mid_line_C_AB()

        lines = {}
        lines["line A-BC"] = line_A_BC
        lines["line B-AC"] = line_B_AC
        lines["line C-AB"] = line_C_AB
        return lines

    # get the mid line equation format of C_AB
    def get_mid_line_string_C_AB(self):
        line = self.get_mid_line_C_AB()
        return self.get_line_string(line)

    # get the mid line equation format of B-AC
    def get_mid_line_string_B_AC(self):
        line = self.get_mid_line_B_AC()
        return self.get_line_string(line)

    # get the mid line equation format of A-BC
    def get_mid_line_string_A_BC(self):
        line = self.get_mid_line_A_BC()
        return self.get_line_string(line)

    # get the mid line equation format information
    # of the triangle
    def get_mid_line_strings(self):
        line_A_BC = self.get_mid_line_string_A_BC()
        line_B_AC = self.get_mid_line_string_B_AC()
        line_C_AB = self.get_mid_line_string_C_AB()

        lines = {}
        lines["line A-BC"] = line_A_BC
        lines["line B-AC"] = line_B_AC
        lines["line C-AB"] = line_C_AB
        return lines

    # get the determinant of a 2D matrix
    # matrix[0] matrix[1]
    # matrix[2] matrix[3]
    def get_determinant(self,matrix):
        return matrix[0]*matrix[3] - matrix[1]*matrix[2]

    # get the intersect point of two lines
    def get_intersect_point(self,line1,line2):
        # get A
        matrix = []
        matrix.append(line1[0])
        matrix.append(line1[1])
        matrix.append(line2[0])
        matrix.append(line2[1])
        A = self.get_determinant(matrix)
        if A == 0:
            print("lines are parallel to each other")
            return -1
        # get B
        matrix = []
        matrix.append(line1[2])
        matrix.append(line1[1])
        matrix.append(line2[2])
        matrix.append(line2[1])
        B = self.get_determinant(matrix)

        # get C
        matrix = []
        matrix.append(line1[0])
        matrix.append(line1[2])
        matrix.append(line2[0])
        matrix.append(line2[2])
        C = self.get_determinant(matrix)

        # get x,y
        x = - B / A
        y = - C / A

        return [x,y]

    # test the get_intersect_point function
    def test_get_intersect_point(self):
        midLines = self.get_mid_lines()
        line_A_BC = midLines["line A-BC"]
        line_B_AC = midLines["line B-AC"]
        line_C_AB = midLines["line C-AB"]

        print(self.get_intersect_point(line_A_BC,line_B_AC))
        print(self.get_intersect_point(line_B_AC,line_A_BC))
        print(self.get_intersect_point(line_B_AC,line_C_AB))
        print(self.get_intersect_point(line_C_AB,line_B_AC))
        print(self.get_intersect_point(line_A_BC,line_C_AB))
        print(self.get_intersect_point(line_C_AB,line_A_BC))
        print("gravity center",self.get_gravity_center())

