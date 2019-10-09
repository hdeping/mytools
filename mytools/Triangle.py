#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-08 12:25:35
    @project      : some toolkits for triangle computation
    @version      : 0.1
    @source file  : Triangle.py

============================
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt






class Triangle():
    """docstring for Triangle
    There are a set of tools to deal with triangles, to get
    the all kinds of properties of a triangle, you can get 
    area, perimeter, radius or center of the inscribed circle,
    circumscribed circle or escribed circles. You can get 
    lengths, angles, cosine values of the angles. Also, you can
    get the equations of orthogonal lines, middle lines, middle
    orthogonal lines or angular bisectors.
    """
    def __init__(self):
        """
        In this module, a point is denoted by [a,b],
        and a line is denoted by [A,B,C]
        Parameters used in this module are listed here:
        self.vertices:
            three vertices of the triangle
        self.lengths:
            lengths of the three sides of the triangle
        self.angles:
            three angles (0-180) of the triangle
        self.cosines:
            cosine values of the three angles
        self.area:
            the area of the triangle
        self.sideLines:
            three sides of the triangle
        self.orthoLines:
            three orthogal lines of the triangle
        self.midLines:
            three middle lines of the triangle
        self.midOrthoLines:
            three middle orthogonal lines of the triangle
        self.bisectLines:
            three inner angular bisectors  
            and three outer angular bisectors
        self.insCenter:
            the center of the inscribed circle of the triangle
        self.insRadius:
            the radius of the inscribed circle of the triangle
        self.esCenters:
            three centers of the escribed circle of the triangle
        self.esRadii:
            three radii of the escribed circle of the triangle
        self.orthoCenter:
            the orthogonal center of the triangle
        self.weightCenter:
            the weight center of the triangle
        self.circumRadius:
            the radius of the circumscribed circle
        self.circumCenter:
            the center of the circumscribed circle
        self.orthoPoints:
            three orthogonal points of the triangle
        self.orders:
            indeces of the three vertix-vertix pairs
        """
        super(Triangle, self).__init__()
        vertices = np.array([[3,0],[0,4],[0,0]])
        self.vertices = vertices
        self.lengths  = None
        self.angles   = None 
        self.cosines  = None 
        self.area     = None

        self.sideLines     = None
        self.orthoLines    = None 
        self.midLines      = None 
        self.midOrthoLines = None 
        self.orthoLines    = None
        self.bisectLines   = None

        self.insCenter    = None 
        self.insRadius    = None 
        self.esCenters    = None 
        self.esRadius     = None 
        self.orthoCenter  = None 
        self.weightCenter = None
        self.circumRadius = None
        self.circumCenter = None

        self.orthoPoints = None
        self.orthoCenter = None
        self.orders      = [[0,1],[0,2],[1,2]]

    def setVertices(self,vertices):
        """
        setup for the coordinate of vertices
        """
        print("set vertices to ",vertices)
        
        vertices = np.array(vertices)
        self.vertices = vertices
        self.printVertices()
        return 
    def printVertices(self):
        """
        print out three vertices of the triangle
        """
        labels = ["A","B","C"]
        prefix = "vertix"
        array  = self.vertices
        self.printArray(prefix, labels, array)

        return 
    def printArray(self,prefix,labels,array):
        """
        input: prefix, such as "length"
        input: labels, such as ["A"]
        array: array type, such as [1] or [[1,2]]
        """
        number = len(array)
        assert len(labels) == number
        for i in range(number):
            print("%s %s: "%(prefix,labels[i]),array[i])

        return
    def getLengths(self):
        """
        get three side length of the triangle
        """
        self.lengths = []
        
        for i,j in self.orders:
            v1 = self.vertices[i]
            v2 = self.vertices[j]
            length = self.getDist(v1,v2)
            self.lengths.append(length)
            
        return
    def printLengths(self):
        """
        print out three lengths of the triangle
        """
        labels = ["AB","AC","BC"]
        prefix = "length"
        array  = self.lengths
        self.printArray(prefix, labels, array)


    def getDist(self,vertix1,vertix2):
        """
        get the L2 distance of two vertices
        """
        vertix = vertix1 - vertix2
        dist = np.linalg.norm(vertix)
        return dist
    
    def isTriangle(self):
        """
        judge if it is a triangle
        a + b > c -> (a+b+c)/2 > c
        """
        lengths = np.array(self.lengths)
        p       = sum(lengths)/2 
        if sum(lengths < p) == 3:
            return True
        else:
            print("%.2f,%.2f,%.2f can not construct a triangle"%(tuple(lengths)))
            return False

    def getArea(self):
        """
        get the area of the triangle
        """
        if self.isTriangle():
            p = sum(self.lengths)/2 
            area = p 
            for i in range(3):
                area = area * (p - self.lengths[i])
            self.area = area**0.5
            return
        else:
            return 
    def getVerticesLine(self,vertix1,vertix2):
        """
        (x1,y1), (x2,y2) 
        (y - y1)/(y2 - y1) = (x - x1)/(x2 - x1)
        (y2 - y1)(x - x1) + (x1 - x2)(y - y1)
        input: two vertices
        return: a line [A,B,C]
        """
        A = vertix2[1] - vertix1[1]
        B = vertix1[0] - vertix2[0]
        return self.getVertixSlope(vertix1, [A,B]) 
    def getVertixSlope(self,vertix1,slope):
        """
        get the line with a vertix and slope
        input: vertix [x,y] and a slope [A,B]
        return: a line [A,B,C]
        """
        A = slope[0]
        B = slope[1]
        C = - (A*vertix1[0] + B*vertix1[1])
        line = [A,B,C]
        return line
    def getInterVertix(self,line1,line2):
        """
        get the intersection of two lines
        input: two lines
        return: a point
        """
        data = []
        data.append(line1)   
        data.append(line2)   
        data = np.array(data)

        result = np.linalg.solve(data[:2,:2],-data[:,2])
        return result

    def getCircumCenter(self):
        """
        center of the circum circle
        intersection point of two middle orthogonal lines
        """
        line1 = self.getMidOrthoLine(self.vertices[0],self.vertices[1])
        line2 = self.getMidOrthoLine(self.vertices[0],self.vertices[2])
        point = self.getInterVertix(line1, line2)
        return point

    def getMidPoint(self,vertix1,vertix2):
        """
        middle point of two points
        """
        vertix  = (vertix1 + vertix2)/2 
        return vertix
    def getMidOrthoLine(self,vertix1,vertix2):
        """
        get a middle orthogonal line
        input: two points
        return: a line
        """
        vertix = self.getMidPoint(vertix1,vertix2)
        line = self.getVertixSlope(vertix,vertix2 - vertix1)
        return line

    def getOrthoLine(self,vertix1,vertix2,vertix3):
        """
        get the orthogonal line give three  vertices
        a point vertix3 on the line, 
        which is orthogonal to the one through vertix1 and vertix2 
        """
        line = self.getVertixSlope(vertix3,vertix2 - vertix1)
        return line

    def getOrthoLineByIndex(self,i,j,k):
        """
        get the orthogonal line give three indeces of the vertices
        """
        vertix1 = self.vertices[i]
        vertix2 = self.vertices[j]
        vertix3 = self.vertices[k]
        line = self.getVertixSlope(vertix3,vertix2 - vertix1)
        return line

    def getOrthoLines(self):
        """
        get three orthogonal lines of the triangle
        AA', BB', CC'
        """
        self.orthoLines  = []
        # get lines
        for ii,[i,j] in enumerate(self.orders):
            line = self.getOrthoLineByIndex(i,j,2 - ii)
            self.orthoLines.append(line)
        self.getNewOrder(self.orthoLines)

        return 
    def getMidLines(self):
        """
        get three middle lines
        """
        self.midLines  = []
        # get lines
        # order: 2,1,0
        for ii,[i,j] in enumerate(self.orders):
            vertix = self.getMidPoint(self.vertices[i],self.vertices[j])
            line = self.getVerticesLine(vertix,self.vertices[ii])
            self.midLines.append(line)
        # get new order
        self.getNewOrder(self.midLines)

        return 

    def getSideLines(self):
        """
        get three side lines
        """
        self.sideLines  = []
        # get lines
        # order: 2,1,0
        for ii,[i,j] in enumerate(self.orders):
            line = self.getVerticesLine(self.vertices[i],self.vertices[j])
            self.sideLines.append(line)

        return

    def getMidOrthoLines(self):
        """
        get three middle orthogonal lines
        """
        self.midOrthoLines = []

        for ii,[i,j] in enumerate(self.orders):
            line = self.getMidOrthoLine(self.vertices[i],self.vertices[j])
            self.midOrthoLines.append(line)

        return

    def getCircumCenter(self):
        """
        get the center of the circumscribed circle of the triangle,
        which is the intersection point of two middle orthogonal
        lines
        """
        self.getMidOrthoLines()

        line1 = self.midOrthoLines[0]
        line2 = self.midOrthoLines[1]
        self.circumCenter = self.getInterVertix(line1, line2)

        return

    def getInsCenter(self):
        """
        get the center of the inscribed circle of the triangle
        s = (aA+bB+cC)/2
        p = (a+b+c)/2
        p_r = s/p
        """
        results = 0 
        for i in range(3):
            results += self.lengths[2-i]*self.vertices[i]
        self.insCenter = results / self.getPerimeter()
        return

    def getEsCenters(self):
        """
        get three centers of the escribed circle of the triangle
        s = (aA+bB+cC)/2
        p = (a+b+c)/2
        p_rA = (s - aA)/(p - a)
        p_rB = (s - bB)/(p - b)
        p_rC = (s - cC)/(p - c)

        """
        results = 0 
        self.esCenters = []
        for i in range(3):
            results += self.lengths[2-i]*self.vertices[i]

        results = results / 2 
        p       = self.getPerimeter() / 2 

        for i in range(3):
            # center = (results - self.vertices[i]) / (p - self.lengths[2-i])
            result = self.lengths[2-i]*self.vertices[i]
            center = (results - result) / (p - self.lengths[2-i])
            self.esCenters.append(center)

        return
    def getOrthoPoints(self):
        """
        get three orthogonal points of  the triangle
        """
        self.getOrthoLines()
        self.getSideLines()

        self.orthoPoints = []
        for i in range(3):
            line1 = self.orthoLines[i]
            line2 = self.sideLines[2 - i]
            point = self.getInterVertix(line1, line2)
            self.orthoPoints.append(point)

        return

    def getOrthoCenter(self):
        """
        get the orthogonal center of the triangle
        """
        self.getOrthoLines()
        line1 = self.orthoLines[0]
        line2 = self.orthoLines[1]
        self.orthoCenter = self.getInterVertix(line1, line2)
        return
    def getNewOrder(self,array):
        """
        get the inverse order of a array
        with a length 3
        """
        array[0],array[-1] = array[-1],array[0]
        return
    def getWeightPoint(self):
        """
        weight point of the triangle
        """
        point = np.average(self.vertices,axis=0)
        self.weightCenter = point
        return 

    def getCosines(self):
        """
        get three cosine values of the angles
        """
        self.cosines = []
        for i,j in self.orders:
            cosine = self.getCosineByIndex(i,j)
            self.cosines.append(cosine)
        return
    
    def getCosineByIndex(self,i,j):
        """
        get cosine values give the indeces of vertices
        input: indeces i and j, the third one should be
                3 - i - j
        return: the cosine value of the angle
        """
        a = self.lengths[i]
        b = self.lengths[j]
        c = self.lengths[3 - i - j]
        result = (a*a + b*b - c*c)/(2*a*b)
        return result

    def getInsRadius(self):
        """
        get the radius of the inscribed circle of the triangle
        """
        self.getArea()
        p = self.getPerimeter() / 2 
        self.insRadius = self.area / p 
        return 
    def getEsRadii(self):
        """
        get three radii of the escribed circle of the triangle
        """
        self.getArea()
        p = self.getPerimeter() / 2 
        self.esRadius = []

        for i in range(3):
            radius = self.area / (p - self.lengths[i])
            self.esRadius.append(radius)

        self.getNewOrder(self.esRadius)
        return 

    def getBisectLines(self):
        """
        """
        self.getInsCenter()
        self.getEsCenters()
        # three inscribed lines 
        # and three escribed lines
        self.bisectLines = []

        # inscribed lines
        for i in range(3):
            line = self.getVerticesLine(self.vertices[i],self.insCenter)
            self.bisectLines.append(line)

        # inscribed lines
        for i in range(3):
            ii = self.orders[2 - i][0]
            jj = self.orders[2 - i][1]
            line = self.getVerticesLine(self.esCenters[ii],self.esCenters[jj])
            self.bisectLines.append(line)

        return

    def getCircumRadius(self):
        """
        """
        self.getArea()
        result = 1/4/self.area
        for i in range(3):
            result = result * self.lengths[i]

        self.circumRadius = result
        return 


    def getPerimeter(self):
        """
        """
        result = sum(self.lengths)
        return result

    def test(self):
        """
        test the module 
        and draw the triangle, inscribed center
        and three escribed centers
        """
        self.setVertices([[3,0],[0,4],[0,0]])
        # self.printVertices()
        self.getLengths()
        # self.lengths = [2,2,2]
        self.printLengths()
        self.isTriangle()
        self.getArea()
        print(self.area)

        line1 = self.getVerticesLine(self.vertices[0],self.vertices[1])
        print(line1)
        line2 = self.getVerticesLine(self.vertices[0],self.vertices[2])
        print(line2)
        A = self.getInterVertix(line1,line2)
        print(A)

        self.getCosines()
        print(self.cosines)
        
        print("外心",self.getWeightPoint())

        self.getInsCenter()
        print("内心",self.insCenter)

        self.getCircumCenter()
        self.getOrthoPoints()
        self.getEsCenters()        
        
        print(self.esCenters)
        print("边的直线方程",self.sideLines)

        self.getBisectLines()
        self.testBisect()

        self.getInsRadius()
        self.getEsRadius()
        print(self.insRadius,self.esRadius)
        self.getEsRadii()
        print(self.insRadius,self.esRadii)
        print("旁心",self.esCenters)
        self.draw()
    def testBisect(self):
        """
        test the bisection lines.
        judge if the inner bisection lines 
        are orthogonal to the outer ones.
        There are three orthogonal pairs.
        """
        # print("角平分线",self.bisectLines)
        for line in self.bisectLines:
            print(line)
        for i in range(3):
            line1 = self.bisectLines[i]
            line2 = self.bisectLines[i+3]
            self.isOrtho(line1,line2)

        return

    def draw(self):
        """
        draw the triangle, inscribed center
        and three escribed centers
        """
        ax = plt.gca()
        ax.set_aspect(1)
        self.esCenters = np.array(self.esCenters)
        plt.plot(self.vertices[:,0],self.vertices[:,1])
        plt.plot(self.esCenters[:,0],self.esCenters[:,1])
        self.drawLine(self.esCenters[0],self.esCenters[-1])
        # plt.plot(x,y,"o")
        self.drawLine(self.vertices[0],self.vertices[-1])
        for i in range(3):
            self.drawLine(self.insCenter,self.vertices[i])
        plt.savefig("triangle.png",dpi=200)
        plt.show()
        return
    def drawLine(self,point1,point2):
        """
        draw the line with two points
        input: two points
        """
        data = []
        data.append(point1)
        data.append(point2)
        data = np.array(data)
        plt.plot(data[:,0],data[:,1])

        return

    def isOrtho(self,line1,line2):
        """
        """
        vec1 = np.array(line1[:2])
        vec2 = np.array(line2[:2])
        res = np.dot(vec1,vec2)
        print("%.2f,%.2f,%.2f,%.2f,%.2f"%(*tuple(vec1),*tuple(vec2),res))
        
        if res == 0:
            print("IS ORTHOGOGAL")
            
            return True
        else:
            print("NOT ORTHOGOGAL !!!")
            
            return False

    def getVerticeAngle(self,vertex1,vertex2,vertex3):
        """
        """
        vec1 = vertex1 - vertex2
        vec2 = vertex3 - vertex2
        angle = self.getVectorAngle(vec1,vec2)
        print("angle",angle)
        
        return
    def getVectorAngle(self,vec1,vec2):
        """
        """
        angle = np.dot(vec1,vec2)
        angle = angle/np.linalg.norm(vec1)
        angle = angle/np.linalg.norm(vec2)
        return angle
    def printTriInfo(self):
        """
        """
        print("vertices: ",self.vertices)
        print("lengths: ",self.lengths)
        print("angles: ",self.angles)
        print("cosines: ",self.cosines)
        print("area: ",self.area)
        print("sideLines: ",self.sideLines)
        print("orthoLines: ",self.orthoLines)
        print("midLines: ",self.midLines)
        print("midOrthoLines: ",self.midOrthoLines)
        print("orthoLines: ",self.orthoLines)
        print("bisectLines: ",self.bisectLines)
        print("insCenter: ",self.insCenter)
        print("insRadius: ",self.insRadius)
        print("esCenters: ",self.esCenters)
        print("esRadius: ",self.esRadius)
        print("orthoCenter: ",self.orthoCenter)
        print("weightCenter: ",self.weightCenter)
        print("circumRadius: ",self.circumRadius)
        print("circumCenter: ",self.circumCenter)
        print("orthoPoints: ",self.orthoPoints)

        return

    def getAngles(self):
        """
        """
        self.getCosines()
        self.angles = []
        for value in self.cosines:
            angle = np.arccos(value)*180/np.pi
            self.angles.append(angle)

        return

    def run(self):
        """
        """
        self.setVertices([[3,0],[0,4],[0,1]])
        self.getLengths()
        self.getAngles()
        self.getArea()
        self.getOrthoCenter()
        self.getOrthoPoints()
        self.getCircumCenter()
        self.getBisectLines()
        self.getInsRadius()
        self.getEsRadii()
        self.getWeightPoint()
        self.getCircumRadius()
        self.getOrthoCenter()
        self.getMidLines()

        self.printTriInfo()
        self.testBisect()
        self.draw()              
