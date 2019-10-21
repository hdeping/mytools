#!/usr/local/bin/python3

from base import base

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
