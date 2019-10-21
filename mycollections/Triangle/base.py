#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt

class base():
    # init function
    def __init__(self,points):
        self.A = points[0]
        self.B = points[1]
        self.C = points[2]
        self.points = points

    # get all the coordinates of points 
    def get_points(self):
        return self.points

    # get all the coordinates of A
    def get_point_A(self):
        return self.A

    # get all the coordinates of B
    def get_point_B(self):
        return self.B

    # get all the coordinates of C
    def get_point_C(self):
        return self.C

    # get the distance of two points
    def get_lateral(self,coor1,coor2):
        # get the lateral length
        length = coor2 - coor1
        length = length**2
        length = length.sum()
        length = np.sqrt(length)
        return length

    # get the length of AB
    def get_lateral_AB(self):
        return self.get_lateral(self.A,self.B)

    # get the length of AC
    def get_lateral_AC(self):
        return self.get_lateral(self.A,self.C)

    # get the length of BC
    def get_lateral_BC(self):
        return self.get_lateral(self.B,self.C)

    # side length information of the triangle
    def get_laterals(self):
        laterals   = {}
        a = self.get_lateral_BC()
        b = self.get_lateral_AC()
        c = self.get_lateral_AB()
        laterals["BC"] = a
        laterals["AC"] = b
        laterals["AB"] = c
        
        return laterals

    # get the angle with three sides
    # the output will range from 0 to 180
    def get_angle(self,a,b,c):
        a2 = a**2
        b2 = b**2
        c2 = c**2
        cosValue = (a2+b2-c2)/(2*a*b)
        angle = np.arccos(cosValue)
        angle = angle*180.0/np.pi
        return angle

    # get the angle of A
    def get_angle_A(self):
        laterals = self.get_laterals()
        a = laterals["BC"]
        b = laterals["AC"]
        c = laterals["AB"]
        angle = self.get_angle(c,b,a)
        return angle

    # get the angle of B
    def get_angle_B(self):
        laterals = self.get_laterals()
        a = laterals["BC"]
        b = laterals["AC"]
        c = laterals["AB"]
        angle = self.get_angle(a,c,b)
        return angle

    # get the angle of C
    def get_angle_C(self):
        laterals = self.get_laterals()
        a = laterals["BC"]
        b = laterals["AC"]
        c = laterals["AB"]
        angle = self.get_angle(a,b,c)
        return angle

    # get the angle information of the triangle
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

    # get line equation of two points
    def get_line(self,coor1,coor2):
        # a*x + b*y + c = 0
        a = coor2[1] - coor1[1]
        b = coor1[0] - coor2[0]
        c = -(a*coor1[0] + b*coor1[1])
        return [a,b,c]

    # get line equation of AB
    def get_line_AB(self):
        return self.get_line(self.A,self.B)

    # get line equation of AC
    def get_line_AC(self):
        return self.get_line(self.A,self.C)

    # get line equation of BC
    def get_line_BC(self):
        return self.get_line(self.B,self.C)

    # get the line equation information of the
    # triangle
    def get_lines(self):
        line_AB = self.get_line_AB()
        line_AC = self.get_line_AC()
        line_BC = self.get_line_BC()

        lines = {}
        lines["line AB"] = line_AB
        lines["line AC"] = line_AC
        lines["line BC"] = line_BC
        return lines

    # get the "ax+by+c=0" format of a
    # line equation
    def get_line_string(self,line):
        # inverse the line
        if line[0] < 0:
            for i in range(3):
                line[i] = - line[i]

        a = abs(line[0])
        b = abs(line[1])
        c = abs(line[2])
        string1 = "%.2fx + %.2fy + %.2f = 0"%(a,b,c)
        string2 = "%.2fx + %.2fy - %.2f = 0"%(a,b,c)
        string3 = "%.2fx - %.2fy + %.2f = 0"%(a,b,c)
        string4 = "%.2fx - %.2fy - %.2f = 0"%(a,b,c)

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

    # get the line equation format of AB
    def get_line_string_AB(self):
        line_AB = self.get_line_AB()
        return self.get_line_string(line_AB)

    # get the line equation format of AC
    def get_line_string_AC(self):
        line_AC = self.get_line_AC()
        return self.get_line_string(line_AC)
    
    # get the line equation format of BC
    def get_line_string_BC(self):
        line_BC = self.get_line_BC()
        return self.get_line_string(line_BC)

    # get the line equation format information
    # of the triangle
    def get_line_strings(self):
        line_AB = self.get_line_string_AB()
        line_AC = self.get_line_string_AC()
        line_BC = self.get_line_string_BC()
        lines = {}
        lines["line AB"] = line_AB
        lines["line AC"] = line_AC
        lines["line BC"] = line_BC
        return lines

    # get the area of the triangle
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

    # judge if the three points would
    # construct a triangle
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

    #draw a line with two points as inputs
    def draw_line(self,point1,point2):
        x  = []
        x.append(point1[0])
        x.append(point2[0])
        y  = []
        y.append(point1[1])
        y.append(point2[1])
        plt.plot(x,y,'b','-o',linewidth=4)

    # draw a circle with the center and radius as inputs
    def draw_circle(self,center,radius):
        theta = np.linspace(0,2*np.pi,100)
        x     = radius*np.cos(theta) + center[0]
        y     = radius*np.sin(theta) + center[1]
        plt.plot(x,y,'#FF00FF','-',linewidth = 4)

    # draw the triangle
    def draw(self):
        plt.figure(figsize = (9,9))
        plt.axis("equal")
        self.draw_line(self.A,self.B)
        self.draw_line(self.A,self.C)
        self.draw_line(self.C,self.B)
        radius = self.get_lateral_AB()
        #self.draw_circle(self.A,radius)
        #self.draw_circle(self.B,radius)
        plt.show()
        return 1

    # get the  radius of the circum circle
    # which contains the three points
    # of the triangle
    def get_circum_radius(self):
        laterals = self.get_laterals()
        a = laterals["AB"]
        b = laterals["AC"]
        c = laterals["BC"]
        area = self.get_area()
        radius = a*b*c/(4.0*area)
        return radius

    # get the  radius of the inscribed circle
    # which is tangent to the three sides 
    # of the triangle
    def get_inscribe_radius(self):
        laterals = self.get_laterals()
        a = laterals["AB"]
        b = laterals["AC"]
        c = laterals["BC"]
        p = (a+b+c)/2
        area = self.get_area()
        radius = area/p
        return radius

    # get the radius information of the triangle
    def get_radiuses(self):
        laterals = self.get_laterals()
        a = laterals["BC"]
        b = laterals["AC"]
        c = laterals["AB"]
        p = (a+b+c)/2
        p_a = p - a
        p_b = p - b
        p_c = p - c
        area = self.get_area()
        circum_radius    = a*b*c/(4.0*area)
        inscribe_radius  = area/p
        escribe_radius_a = area/p_a
        escribe_radius_b = area/p_b
        escribe_radius_c = area/p_c
        radiuses = {}
        radiuses["circum radius"] = circum_radius
        radiuses["inscribe radius"] = inscribe_radius
        radiuses["escribe radius A"] = escribe_radius_a
        radiuses["escribe radius B"] = escribe_radius_b
        radiuses["escribe radius C"] = escribe_radius_c
        return radiuses

    # get the radius of the escribed circle A
    # which is tangent to the line BC
    def get_escribe_radius_A(self):
        laterals = self.get_laterals()
        a = laterals["BC"]
        b = laterals["AC"]
        c = laterals["AB"]
        p = (a+b+c)/2
        p = p - a
        area = self.get_area()
        radius = area/p
        return radius

    # get the radius of the escribed circle B
    # which is tangent to the line AC
    def get_escribe_radius_B(self):
        laterals = self.get_laterals()
        a = laterals["BC"]
        b = laterals["AC"]
        c = laterals["AB"]
        p = (a+b+c)/2
        p = p - b
        area = self.get_area()
        radius = area/p
        return radius

    # get the radius of the escribed circle C
    # which is tangent to the line AB
    def get_escribe_radius_C(self):
        laterals = self.get_laterals()
        a = laterals["BC"]
        b = laterals["AC"]
        c = laterals["AB"]
        p = (a+b+c)/2
        p = p - c
        area = self.get_area()
        radius = area/p
        return radius

