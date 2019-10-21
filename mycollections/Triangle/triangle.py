#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt

class Triangle():
    def __init__(self,points):
        self.A = points[0]
        self.B = points[1]
        self.C = points[2]
        self.points = points
    def get_points(self):
        return self.points
    def get_point_A(self):
        return self.A
    def get_point_B(self):
        return self.B
    def get_point_C(self):
        return self.C
    def get_lateral(self,coor1,coor2):
        # get the lateral length
        length = coor2 - coor1
        length = length**2
        length = length.sum()
        length = np.sqrt(length)
        return length
    def get_lateral_AB(self):
        return self.get_lateral(self.A,self.B)
    def get_lateral_AC(self):
        return self.get_lateral(self.A,self.C)
    def get_lateral_BC(self):
        return self.get_lateral(self.B,self.C)
    def get_laterals(self):
        laterals   = {}
        lateral_AB = self.get_lateral_AB()
        lateral_AC = self.get_lateral_AC()
        lateral_BC = self.get_lateral_BC()
        laterals["AB"] = lateral_AB
        laterals["AC"] = lateral_AC
        laterals["BC"] = lateral_BC
        
        return laterals
    def get_angle(self,length_a,length_b,length_c):
        a2 = length_a**2
        b2 = length_b**2
        c2 = length_c**2
        cosValue = (a2+b2-c2)/(2*length_a*length_b)
        angle = np.arccos(cosValue)
        angle = angle*180.0/np.pi
        return angle
    def get_angle_A(self):
        laterals = self.get_laterals()
        length_a = laterals["AB"]
        length_b = laterals["AC"]
        length_c = laterals["BC"]
        angle = self.get_angle(length_a,length_b,length_c)
        return angle
    def get_angle_B(self):
        laterals = self.get_laterals()
        length_a = laterals["AB"]
        length_b = laterals["BC"]
        length_c = laterals["AC"]
        angle = self.get_angle(length_a,length_b,length_c)
        return angle
    def get_angle_C(self):
        laterals = self.get_laterals()
        length_a = laterals["AC"]
        length_b = laterals["BC"]
        length_c = laterals["AB"]
        angle = self.get_angle(length_a,length_b,length_c)
        return angle
        
    def get_angles(self):
        laterals = self.get_laterals()
        length_AC = laterals["AC"]
        length_BC = laterals["BC"]
        length_AB = laterals["AB"]
        angle_A = self.get_angle(length_AC,length_AB,length_BC)
        angle_B = self.get_angle(length_AB,length_BC,length_AC)
        angle_C = self.get_angle(length_AC,length_BC,length_AB)
        angles = {}
        angles["angle A"] = angle_A
        angles["angle B"] = angle_B
        angles["angle C"] = angle_C

        return angles
    def get_line(self,coor1,coor2):
        # a*x + b*y + c = 0
        a = coor2[1] - coor1[1]
        b = coor1[0] - coor2[0]
        c = -(a*coor1[0] + b*coor1[1])
        return [a,b,c]
    def get_line_AB(self):
        return self.get_line(self.A,self.B)
    def get_line_AC(self):
        return self.get_line(self.A,self.C)
    def get_line_BC(self):
        return self.get_line(self.B,self.C)
    def get_lines(self):
        line_AB = self.get_line_AB()
        line_AC = self.get_line_AC()
        line_BC = self.get_line_BC()

        lines = {}
        lines["line AB"] = line_AB
        lines["line AC"] = line_AC
        lines["line BC"] = line_BC
        return lines
    def get_line_string(self,line):
        # inverse the line
        if line[0] < 0:
            for i in range(3):
                line[i] = - line[i]

        values = [abs(line[0]),abs(line[1]),abs(line[2])]
        string1 = "%.2fx + %.2fy + %.2f = 0"%(values[0],values[1],values[2])
        string2 = "%.2fx + %.2fy - %.2f = 0"%(values[0],values[1],values[2])
        string3 = "%.2fx - %.2fy + %.2f = 0"%(values[0],values[1],values[2])
        string4 = "%.2fx - %.2fy - %.2f = 0"%(values[0],values[1],values[2])
        if line[1] > 0:
            if line[2] > 0:
                return string1
            else:
                return string2
        else:
            if line[2] > 0:
                return string3
            else:
                return string4
    def get_line_string_AB(self):
        line_AB = self.get_line_AB()
        return self.get_line_string(line_AB)
    def get_line_string_AC(self):
        line_AC = self.get_line_AC()
        return self.get_line_string(line_AC)
    def get_line_string_BC(self):
        line_BC = self.get_line_BC()
        return self.get_line_string(line_BC)
    def get_line_strings(self):
        line_AB = self.get_line_string_AB()
        line_AC = self.get_line_string_AC()
        line_BC = self.get_line_string_BC()
        lines = {}
        lines["line AB"] = line_AB
        lines["line AC"] = line_AC
        lines["line BC"] = line_BC
        return lines
    def get_area(self):
        if not self.isTriangle():
            print("There is no area")
            return 0
            
        laterals = self.get_laterals()
        a = laterals["AB"]
        b = laterals["AC"]
        c = laterals["BC"]
        p = (a+b+c)/2
        p1 = p - a
        p2 = p - b
        p3 = p - c
        area = np.sqrt(p*p1*p2*p3)
        return area
    def isTriangle(self):
        laterals = self.get_laterals()
        a = laterals["AB"]
        b = laterals["AC"]
        c = laterals["BC"]
        p1 = (a + b > c)
        p2 = (a + c > b)
        p3 = (c + b > a)
        if p1 and p2 and p3:
            print("Yes, It is a triangle")
            return True
        else:
            print("No, It is not a triangle")
            return False
    def draw(self):
        points = self.points
        plt.figure(figsize = (9,9))
        plt.axis("equal")
        x = points[:2,0]
        y = points[:2,1]
        plt.plot(x,y,'b','-o',linewidth=4)
        x = points[1:,0]
        y = points[1:,1]
        plt.plot(x,y,'b','-o',linewidth=4)
        x = [points[0,0],points[-1,0]]
        y = [points[0,1],points[-1,1]]
        plt.plot(x,y,'b','-o',linewidth=4)
        plt.show()
        return 1
    def get_circum_radius():
        laterals = self.get_laterals()
        a = laterals["AB"]
        b = laterals["AC"]
        c = laterals["BC"]
        area = self.get_area()
        radius = a*b*c/(4.0*area)
        return radius
    def get_inscribe_radius(self):
        laterals = self.get_laterals()
        a = laterals["AB"]
        b = laterals["AC"]
        c = laterals["BC"]
        p = (a+b+c)/2
        area = self.get_area()
        radius = area/p
        return radius
    def get_radiuses(self):
        laterals = self.get_laterals()
        a = laterals["AB"]
        b = laterals["AC"]
        c = laterals["BC"]
        p = (a+b+c)/2
        area = self.get_area()
        radius1 = a*b*c/(4.0*area)
        radius2 = area/p
        return [radius1,radius2]


