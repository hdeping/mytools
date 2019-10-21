#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-21 13:32:30
    @project      : Triangle project
    @version      : 1.0
    @source file  : Triangle.py

============================
"""


from .TriBase import TriBase
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class Triangle(TriBase):
    """
    all kinds of methods for triangle computation
    """
    def __init__(self,points):
        self.A = points[0]
        self.B = points[1]
        self.C = points[2]
        self.points = points
        return
    
    def get_mid_point(self,coor1,coor2):
        """
        get mid point of two points
        """
        x = (coor1[0] + coor2[0])/2
        y = (coor1[1] + coor2[1])/2
        res = [x,y]
        return res

    def get_mid_point_AB(self):
        """
        get mid point of AB
        """
        return self.get_mid_point(self.A,self.B)

    def get_mid_point_AC(self):
        """
        get mid point of AC
        """
        return self.get_mid_point(self.A,self.C)

    def get_mid_point_BC(self):
        """
        get mid point of BC
        """
        return self.get_mid_point(self.C,self.B)

    def get_mid_points(self):
        """
        mid point information of the triangle
        """
        point_AB = self.get_mid_point_AB()
        point_AC = self.get_mid_point_AC()
        point_BC = self.get_mid_point_BC()
        points = {}
        points["AB"] = point_AB
        points["AC"] = point_AC
        points["BC"] = point_BC
        return points

    def get_gravity_center(self):
        """
        get the gravity center
        return:
            center, array, coordinate of the gravity center
        """
        center = np.mean(self.points,axis = 0)
        center = list(center)
        return center

    def draw_mid_lines(self):
        """
        draw the mid lines
        """
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

    def get_mid_line_A_BC(self):
        """
        get the mid line of A-BC
        """
        point_BC = self.get_mid_point_BC()
        return self.get_line(self.A,point_BC)

    def get_mid_line_B_AC(self):
        """
        get the mid line of B-AC
        """
        point_AC = self.get_mid_point_AC()
        return self.get_line(self.B,point_AC)

    def get_mid_line_C_AB(self):
        """
        get the mid line of C-AB
        """
        point_AB = self.get_mid_point_AB()
        return self.get_line(self.C,point_AB)

    def get_mid_lines(self):
        """
        get the mid line information of the
        triangle
        """
        line_A_BC = self.get_mid_line_A_BC()
        line_B_AC = self.get_mid_line_B_AC()
        line_C_AB = self.get_mid_line_C_AB()

        lines = {}
        lines["line A-BC"] = line_A_BC
        lines["line B-AC"] = line_B_AC
        lines["line C-AB"] = line_C_AB
        return lines

    def get_mid_line_string_C_AB(self):
        """
        get the mid line equation format of C_AB
        """
        line = self.get_mid_line_C_AB()
        return self.get_line_string(line)

    def get_mid_line_string_B_AC(self):
        """
        get the mid line equation format of B-AC
        """
        line = self.get_mid_line_B_AC()
        return self.get_line_string(line)

    def get_mid_line_string_A_BC(self):
        """
        get the mid line equation format of A-BC
        """
        line = self.get_mid_line_A_BC()
        return self.get_line_string(line)

    def get_mid_line_strings(self):
        """
        get the mid line equation format information
        of the triangle
        """
        line_A_BC = self.get_mid_line_string_A_BC()
        line_B_AC = self.get_mid_line_string_B_AC()
        line_C_AB = self.get_mid_line_string_C_AB()

        lines = {}
        lines["line A-BC"] = line_A_BC
        lines["line B-AC"] = line_B_AC
        lines["line C-AB"] = line_C_AB
        return lines

    def get_determinant(self,matrix):
        """
        get the determinant of a 2D matrix
        matrix[0] matrix[1]
        matrix[2] matrix[3]
        """
        return matrix[0]*matrix[3] - matrix[1]*matrix[2]

    def get_intersect_point(self,line1,line2):
        """
        get the intersect point of two lines
        """
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

    def test_get_intersect_point(self):
        """
        test the get_intersect_point function
        """
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

    def get_ortho_line(self,line,point):
        """
        get ortho line of a point and a line
        the ortho line is perpendicular to the input line
        """
        res = []
        res.append(line[1])
        res.append(- line[0])
        c = - (res[0]*point[0] + res[1]*point[1])
        res.append(c)
        return res

    def get_ortho_line_A_BC(self):
        """
        get the ortho line of A-BC
        """
        line = self.get_line_BC()
        return self.get_line(line,self.A)

    def get_ortho_line_B_AC(self):
        """
        get the ortho line of B-AC
        """
        line = self.get_line_AC()
        return self.get_line(line,self.B)

    def get_ortho_line_C_AB(self):
        """
        get the ortho line of C-AB
        """
        line = self.get_line_AB()
        return self.get_line(line,self.C)

    def get_ortho_lines(self):
        """
        get the ortho line information of the
        triangle
        """
        line_A_BC = self.get_ortho_line_A_BC()
        line_B_AC = self.get_ortho_line_B_AC()
        line_C_AB = self.get_ortho_line_C_AB()

        lines = {}
        lines["line A-BC"] = line_A_BC
        lines["line B-AC"] = line_B_AC
        lines["line C-AB"] = line_C_AB
        return lines

    def get_ortho_line_string_C_AB(self):
        """
        get the ortho line equation format of C_AB
        """
        line = self.get_ortho_line_C_AB()
        return self.get_line_string(line)

    def get_ortho_line_string_B_AC(self):
        """
        get the ortho line equation format of B-AC
        """
        line = self.get_ortho_line_B_AC()
        return self.get_line_string(line)

    def get_ortho_line_string_A_BC(self):
        """
        get the ortho line equation format of A-BC
        """
        line = self.get_ortho_line_A_BC()
        return self.get_line_string(line)

    def get_ortho_line_strings(self):
        """
        get the ortho line equation format information
        of the triangle
        """
        line_A_BC = self.get_ortho_line_string_A_BC()
        line_B_AC = self.get_ortho_line_string_B_AC()
        line_C_AB = self.get_ortho_line_string_C_AB()

        lines = {}
        lines["line A-BC"] = line_A_BC
        lines["line B-AC"] = line_B_AC
        lines["line C-AB"] = line_C_AB
        return lines

    def get_ortho_point(self,line,point):
        """
        get ortho point of a point and a line
        """
        ortho_line = self.get_ortho_line(line,point)
        #print(ortho_line)
        res = self.get_intersect_point(line,ortho_line)
        return res

    def get_ortho_point_C_AB(self):
        """
        get ortho point of AB
        """
        line = self.get_line_AB()
        return self.get_ortho_point(line,self.C)

    def get_ortho_point_B_AC(self):
        """
        get ortho point of AC
        """
        line = self.get_line_AC()
        return self.get_ortho_point(line,self.B)

    def get_ortho_point_A_BC(self):
        """
        get ortho point of BC
        """
        line = self.get_line_BC()
        return self.get_ortho_point(line,self.A)

    def get_ortho_points(self):
        """
        ortho point information of the triangle
        """
        point_C_AB = self.get_ortho_point_C_AB()
        point_B_AC = self.get_ortho_point_B_AC()
        point_A_BC = self.get_ortho_point_A_BC()
        points = {}
        points["point C-AB"] = point_C_AB
        points["point B-AC"] = point_B_AC
        points["point A-BC"] = point_A_BC
        return points

    def draw_ortho_lines(self):
        """
        draw the orthogonal lines
        """
        plt.figure(figsize = (9,9))
        plt.axis("equal")

        self.draw_line(self.A,self.B)
        self.draw_line(self.A,self.C)
        self.draw_line(self.C,self.B)

        points   = self.get_ortho_points()
        point_C_AB =  points["point C-AB"]
        point_B_AC =  points["point B-AC"]
        point_A_BC =  points["point A-BC"]

        self.draw_line(self.A,point_A_BC)
        self.draw_line(self.B,point_B_AC)
        self.draw_line(self.C,point_C_AB)

        plt.savefig("orthogonal.png",dpi=300)
        plt.show()

    def get_vertical_line(self,point1,point2):
        """
        get vertical line of two points 
        the vertical line is perpendicular to the 
        line constructed by the input points
        """
        mid_point = self.get_mid_point(point1,point2)
        line      = self.get_line(point1,point2)
        res       = self.get_ortho_line(line,mid_point)
        return res

    def get_vertical_line_BC(self):
        """
        get the vertical line of BC
        """
        return self.get_vertical_line(self.B,self.C)

    def get_vertical_line_AC(self):
        """
        get the vertical line of AC
        """
        return self.get_vertical_line(self.C,self.A)

    def get_vertical_line_AB(self):
        """
        get the vertical line of AB
        """
        return self.get_vertical_line(self.B,self.A)

    def get_vertical_lines(self):
        """
        get the vertical line information of the
        triangle
        """
        line_BC = self.get_vertical_line_BC()
        line_AC = self.get_vertical_line_AC()
        line_AB = self.get_vertical_line_AB()

        lines = {}
        lines["line BC"] = line_BC
        lines["line AC"] = line_AC
        lines["line AB"] = line_AB
        return lines

    def get_vertical_line_string_AB(self):
        """
        get the vertical line equation format of AB
        """
        line = self.get_vertical_line_AB()
        return self.get_line_string(line)

    def get_vertical_line_string_AC(self):
        """
        get the vertical line equation format of AC
        """
        line = self.get_vertical_line_AC()
        return self.get_line_string(line)

    def get_vertical_line_string_BC(self):
        """
        get the vertical line equation format of BC
        """
        line = self.get_vertical_line_BC()
        return self.get_line_string(line)

    def get_vertical_line_strings(self):
        """
        get the vertical line equation format information
        of the triangle
        """
        line_BC = self.get_vertical_line_string_BC()
        line_AC = self.get_vertical_line_string_AC()
        line_AB = self.get_vertical_line_string_AB()

        lines = {}
        lines["line BC"] = line_BC
        lines["line AC"] = line_AC
        lines["line AB"] = line_AB
        return lines

    def get_circum_center(self):
        """
        get the center of the circum circle of the 
        triangle
        """
        lines = self.get_vertical_lines()
        line_BC = lines["line BC"] 
        line_AB = lines["line AB"] 
        res = self.get_intersect_point(line_AB,line_BC)
        return res

    def get_ortho_center(self):
        """
        get the ortho center of the 
        triangle
        """
        lines = self.get_ortho_lines()
        line_BC = lines["line A-BC"] 
        line_AB = lines["line C-AB"] 
        res = self.get_intersect_point(line_AB,line_BC)
        return res

    def draw_circum_circle(self):
        """
        draw the circum circle
        """
        plt.figure(figsize = (9,9))
        plt.axis("equal")

        self.draw_line(self.A,self.B)
        self.draw_line(self.A,self.C)
        self.draw_line(self.C,self.B)

        center = self.get_circum_center()
        radius = self.get_circum_radius()
        self.draw_circle(center,radius)

        plt.savefig("circum1.png",dpi=300)
        plt.show()
    
    def get_point_line_value(self,line,point):
        """
        get side value of a line and a point
        """
        res = line[2]
        for i in range(2):
            res += line[i]*point[i]
        return res

    def get_bisector_line(self,point1,point2,point3):
        """
        get bisector line of a triangle
        the bisector line is perpendicular to the 
        line constructed by the input points
        """
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

    def get_inscribe_line_A(self):
        """
        get the inscribe line of A
        """
        line = self.get_bisector_line(self.A,self.B,self.C)
        return line['interior']

    def get_inscribe_line_B(self):
        """
        get the inscribe line of B
        """
        line = self.get_bisector_line(self.B,self.A,self.C)
        return line['interior']

    def get_inscribe_line_C(self):
        """
        get the inscribe line of C
        """
        line =  self.get_bisector_line(self.C,self.A,self.B)
        return line['interior']

    def get_inscribe_lines(self):
        """
        get the inscribe line information of the
        triangle
        """
        line_A = self.get_inscribe_line_A()
        line_B = self.get_inscribe_line_B()
        line_C = self.get_inscribe_line_C()

        lines = {}
        lines["A"] = line_A
        lines["B"] = line_B
        lines["C"] = line_C
        return lines

    def get_inscribe_line_string_A(self):
        """
        get the inscribe line equation format of A
        """
        line = self.get_inscribe_line_A()
        return self.get_line_string(line)

    def get_inscribe_line_string_B(self):
        """
        get the inscribe line equation format of B
        """
        line = self.get_inscribe_line_B()
        return self.get_line_string(line)

    def get_inscribe_line_string_C(self):
        """
        get the inscribe line equation format of C
        """
        line = self.get_inscribe_line_C()
        return self.get_line_string(line)

    def get_inscribe_line_strings(self):
        """
        get the inscribe line equation format information
        of the triangle
        """
        line_A = self.get_inscribe_line_string_A()
        line_B = self.get_inscribe_line_string_B()
        line_C = self.get_inscribe_line_string_C()

        lines = {}
        lines["A"] = line_A
        lines["B"] = line_B
        lines["C"] = line_C
        return lines

    def get_inscribe_center(self):
        """
        get the center of the inscribed circle of the 
        triangle
        """
        lines = self.get_inscribe_lines()
        line_B = lines["B"] 
        line_A = lines["A"] 
        print(line_A,line_B)
        res = self.get_intersect_point(line_A,line_B)
        return res

    def draw_inscribe_circle(self):
        """
        draw the inscribe circle
        """
        plt.figure(figsize = (9,9))
        plt.axis("equal")

        self.draw_line(self.A,self.B)
        self.draw_line(self.A,self.C)
        self.draw_line(self.C,self.B)

        # interior bisector
        inscribe_center = self.get_inscribe_center()
        self.draw_line(self.A,inscribe_center)
        self.draw_line(self.B,inscribe_center)
        self.draw_line(self.C,inscribe_center)

        # inscribe circle
        center = self.get_inscribe_center()
        radius = self.get_inscribe_radius()
        self.draw_circle(center,radius)

        # inscribe circle
        center = self.get_circum_center()
        radius = self.get_circum_radius()
        self.draw_circle(center,radius)
        plt.show()

    def get_escribe_line_A(self):
        """
        get the escribe line of A
        """
        line = self.get_bisector_line(self.A,self.B,self.C)
        return line['exterior']

    def get_escribe_line_B(self):
        """
        get the escribe line of B
        """
        line = self.get_bisector_line(self.B,self.A,self.C)
        return line['exterior']

    def get_escribe_line_C(self):
        """
        get the escribe line of C
        """
        line =  self.get_bisector_line(self.C,self.A,self.B)
        return line['exterior']

    def get_escribe_lines(self):
        """
        get the escribe line information of the
        triangle
        """
        line_A = self.get_escribe_line_A()
        line_B = self.get_escribe_line_B()
        line_C = self.get_escribe_line_C()

        lines = {}
        lines["A"] = line_A
        lines["B"] = line_B
        lines["C"] = line_C
        return lines

    def get_escribe_line_string_A(self):
        """
        get the escribe line equation format of A
        """
        line = self.get_escribe_line_A()
        return self.get_line_string(line)

    def get_escribe_line_string_B(self):
        """
        get the escribe line equation format of B
        """
        line = self.get_escribe_line_B()
        return self.get_line_string(line)

    def get_escribe_line_string_C(self):
        """
        get the escribe line equation format of C
        """
        line = self.get_escribe_line_C()
        return self.get_line_string(line)

    def get_escribe_line_strings(self):
        """
        get the escribe line equation format information
        of the triangle
        """
        line_A = self.get_escribe_line_string_A()
        line_B = self.get_escribe_line_string_B()
        line_C = self.get_escribe_line_string_C()

        lines = {}
        lines["A"] = line_A
        lines["B"] = line_B
        lines["C"] = line_C
        return lines

    def get_escribe_center_A_BC(self):
        """
        get the center of the escribed circle A-BC of the 
        triangle
        """
        lines = self.get_escribe_lines()
        line_B = lines["B"] 
        line_C = lines["C"] 
        res = self.get_intersect_point(line_C,line_B)
        return res

    def get_escribe_center_B_AC(self):
        """
        get the center of the escribed circle B-AC of the 
        triangle
        """
        lines = self.get_escribe_lines()
        line_A = lines["A"] 
        line_C = lines["C"] 
        res = self.get_intersect_point(line_C,line_A)
        return res

    def get_escribe_center_C_AB(self):
        """
        get the center of the escribed circle C-AB of the 
        triangle
        """
        lines = self.get_escribe_lines()
        line_A = lines["A"] 
        line_B = lines["B"] 
        res = self.get_intersect_point(line_B,line_A)
        return res

    def draw_escribe_circle(self):
        """
        draw the circum circle
        """
        plt.figure(figsize = (9,9))
        plt.axis("equal")

        self.draw_line(self.A,self.B)
        self.draw_line(self.A,self.C)
        self.draw_line(self.C,self.B)

        center = self.get_escribe_center_A_BC()
        radius = self.get_escribe_radius_A()
        self.draw_circle(center,radius)
        plt.show()

    def get_centers(self):
        """
        get the centers of the triangle
        gravity center
        ortho center
        circum center
        inscribe center
        3 escribe centers
        """
        gravity_center = self.get_gravity_center()
        circum_center = self.get_circum_center()
        ortho_center = self.get_ortho_center()
        inscribe_center = self.get_inscribe_center()
        escribe_center_A_BC = self.get_escribe_center_A_BC()
        escribe_center_B_AC = self.get_escribe_center_B_AC()
        escribe_center_C_AB = self.get_escribe_center_C_AB()

        centers = {}
        centers["gravity center"]      = gravity_center
        centers["ortho center"]        = ortho_center
        centers["circum center"]       = circum_center
        centers["inscribe center"]     = inscribe_center
        centers["escribe center A-BC"] = escribe_center_A_BC
        centers["escribe center B-AC"] = escribe_center_B_AC
        centers["escribe center C-AB"] = escribe_center_C_AB

        return centers
    
    def get_info(self):
        """
        get all the information of the triangle
        """
        vertices       = self.get_vertices()
        side_lengths   = self.get_laterals()
        angles         = self.get_angles()
        lines          = self.get_line_strings()
        mid_lines      = self.get_mid_line_strings()
        ortho_lines    = self.get_ortho_line_strings()
        vertical_lines = self.get_vertical_line_strings()
        inscribe_lines = self.get_inscribe_line_strings()
        escribe_lines  = self.get_escribe_line_strings()
        radiuses       = self.get_radiuses()
        centers        = self.get_centers()
        area           = self.get_area()

        res = {}
        res['area']             = area
        res['vertices']         = vertices
        res['side lengths']     = side_lengths
        res['angles']           = angles
        res['lines']            = lines         
        res['mid lines']        = mid_lines     
        res['ortho lines']      = ortho_lines   
        res['vertical lines']   = vertical_lines
        res['inscribe lines']   = inscribe_lines
        res['escribe lines']    = escribe_lines 
        res['radiuses']         = radiuses
        res['centers']          = centers

        return res

    def get_extend_points(self,point1,point2):
        """
        extend the line to three times of itself
        get the two end points
        """
        res1 = []
        res2 = []

        res1.append(2*point1[0] - point2[0])
        res1.append(2*point1[1] - point2[1])

        res2.append(2*point2[0] - point1[0])
        res2.append(2*point2[1] - point1[1])
        
        return res1,res2

    def draw_inscribe_circle(self):
        """
        draw all the circles and some lines
        """
        plt.figure(figsize = (9,9))
        plt.axis("equal")

        #plt.xlabel('x',fontsize = 22)
        #plt.xlabel('y',fontsize = 22)

        # lines
        point1,point2 = self.get_extend_points(self.A,self.B)
        self.draw_line(point1,point2)
        point1,point2 = self.get_extend_points(self.A,self.C)
        self.draw_line(point1,point2)
        point1,point2 = self.get_extend_points(self.C,self.B)
        self.draw_line(point1,point2)

        # interior bisector
        inscribe_center = self.get_inscribe_center()
        self.draw_color_line(self.A,inscribe_center,'g')
        self.draw_color_line(self.B,inscribe_center,'g')
        self.draw_color_line(self.C,inscribe_center,'g')


        # inscribe circle
        radiuses = self.get_radiuses()
        centers  = self.get_centers()
        keys = []
        keys.append("circum radius")
        keys.append("inscribe radius")     
        keys.append("escribe radius A-BC") 
        keys.append("escribe radius B-AC") 
        keys.append("escribe radius C-AB") 
        keys.append("circum center")       
        keys.append("inscribe center")     
        keys.append("escribe center A-BC") 
        keys.append("escribe center B-AC") 
        keys.append("escribe center C-AB") 

        # exterior bisector
        center1 = centers[keys[-3]]
        center2 = centers[keys[-2]]
        center3 = centers[keys[-1]]
        self.draw_color_line(center1,center2,'g')
        self.draw_color_line(center1,center3,'g')
        self.draw_color_line(center3,center2,'g')
        for i in range(5):
            center = centers[keys[5+i]]
            radius = radiuses[keys[i]]
            self.draw_circle(center,radius)

        plt.savefig("circles_lines.png",dpi = 300)
        plt.savefig("circles_lines.svg",dpi = 300)
        plt.show()
    def run(self):
        """TODO: Docstring for run.
        :returns: TODO
        entry for the class Triangle
        """
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

        return
