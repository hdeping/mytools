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

    # get ortho line of a point and a line
    # the ortho line is perpendicular to the input line
    def get_ortho_line(self,line,point):
        res = []
        res.append(line[1])
        res.append(- line[0])
        c = - (res[0]*point[0] + res[1]*point[1])
        res.append(c)
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
        #print(ortho_line)
        res = self.get_intersect_point(line,ortho_line)
        return res

    # get ortho point of AB
    def get_ortho_point_C_AB(self):
        line = self.get_line_AB()
        return self.get_ortho_point(line,self.C)

    # get ortho point of AC
    def get_ortho_point_B_AC(self):
        line = self.get_line_AC()
        return self.get_ortho_point(line,self.B)

    # get ortho point of BC
    def get_ortho_point_A_BC(self):
        line = self.get_line_BC()
        return self.get_ortho_point(line,self.A)

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

    # draw the orthogonal lines
    def draw_ortho_lines(self):
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

    # get the ortho center of the 
    # triangle
    def get_ortho_center(self):
        lines = self.get_ortho_lines()
        line_BC = lines["line A-BC"] 
        line_AB = lines["line C-AB"] 
        res = self.get_intersect_point(line_AB,line_BC)
        return res

    # draw the circum circle
    def draw_circum_circle(self):
        plt.figure(figsize = (9,9))
        plt.axis("equal")

        self.draw_line(self.A,self.B)
        self.draw_line(self.A,self.C)
        self.draw_line(self.C,self.B)

        center = self.get_circum_center()
        print("center",center)
        radius = self.get_circum_radius()
        print(radius)
        self.draw_circle(center,radius)

        plt.savefig("circum.png",dpi=300)
        plt.show()
    
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
        print(line_A,line_B)
        res = self.get_intersect_point(line_A,line_B)
        return res

    # draw the circum circle
    def draw_inscribe_circle(self):
        plt.figure(figsize = (9,9))
        plt.axis("equal")

        self.draw_line(self.A,self.B)
        self.draw_line(self.A,self.C)
        self.draw_line(self.C,self.B)

        center = self.get_inscribe_center()
        radius = self.get_inscribe_radius()
        print("center",center)
        print("radius",radius)
        self.draw_circle(center,radius)
        plt.show()

