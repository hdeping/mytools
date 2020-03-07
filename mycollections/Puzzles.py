#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-02-29 17:25:40
    @project      : solutions for math puzzles
    @version      : 1.0
    @source file  : Puzzles.py

============================
"""

from sympy import *
import itertools
import numpy as np


class Puzzles():
    """
    solutions for math puzzles
    """
    def __init__(self):
        super(Puzzles, self).__init__()
    def abSeven(self):
        """
        docstring for abSeven
        """
        arr = np.arange(100)
        lines = itertools.combinations(arr,2)

        num = 7**7
        for (a,b) in lines:
            p1 = (a%7 == 0)
            p2 = (b%7 == 0)
            p3 = ((a+b)%7 == 0)
            if p1 or p2 or p3:
                continue

            target = (a+b)**7 - a**7 - b**7
            if target%num == 0:
                print(a,b,target,num)

        return

    def abcDelta(self):
        """
        docstring for abcDelta
        a,b,c are integers,
        s2 = sqrt(2)
        s3 = sqrt(3)
        abs(a+b*s2+c*s3) < 10^{-11}

        b^2 - 6c^2 = 3, b should be even
        b,c = (3,1)
        """

        solutions = [[1,0],[5,2]]

        for i in range(10):
            arr = []
            for j in range(2):
                num = solutions[-1][j]*10 - solutions[-2][j]
                arr.append(num)
            print(i+3,arr,arr[0]**2 - 6*arr[1]**2)
            solutions.append(arr)
        for i in range(10):
            a = (solutions[i][0] + solutions[i+1][0])//2
            b = (solutions[i][1] + solutions[i+1][1])//2
            print(a,b,a**2-6*b**2)
        for i in range(1000):
            b = 6*i**2 -1 
            b = sqrt(Integer(b))
            if b.is_integer:
                print(b,i)
        
        return


    def nSquare(self):
        """
        docstring for nSquare
        """
        solutions = [[1,0],[2,1]]
        lambda1 = 2*solutions[1][0]
        for i in range(10):
            arr = []
            for j in range(2):
                num = solutions[-1][j]*lambda1 - solutions[-2][j]
                arr.append(num)
            # print(i+2,arr,arr[0]**2 - 7*arr[1]**2)
            solutions.append(arr)
            if i%2 == 0:
                n = 2*arr[0]+2
                print(i,n,sqrt(n))
            
        return

    def fixedPointConic(self):
        """
        docstring for fixedPointConic
        """
        a,b,k,m,l = symbols("a b k m,l")
        x = symbols("x0:3")
        y = symbols("y0:3")

        s = (y[2] - y[0])*(y[1] - y[0])
        s = s - l*((x[2] - x[0])*(x[1] - x[0]))
        print(expand(s))

        # x1*x2, x1 + x2 
        # l = -1
        x12  = ((a*m)**2 - (a*b)**2)/((a*k)**2+b*b)
        x1x2 = -2*a*a*k*m/((a*k)**2+b*b)
        y12  = k*k*x12 + k*m*x1x2 + m*m 
        y1y2 = k*x1x2 + 2*m 
        s = y12 - y[0]*y1y2 + y[0]**2 
        s = s - l*(x12 - x[0]*x1x2 + x[0]**2 )
        # s = s.subs(m,y[0] - k*x[0])
        s = self.simEqn(s,m)

        # print(latex(s))
        
        # xy12 = x1*y2 + x2*y1
        xy12 = 2*k*x12 + m*x1x2

        s = xy12 + 2*x[0]*y[0] - y[0]*x1x2 
        s = s - x[0]*y1y2 
        s = s - l*(x12 - x[0]*x1x2 + x[0]**2)
        # s = s.subs(m,y[0] - k*x[0])
        s = self.simEqn(s,m)
        # s = s.factor()
        s = -s
        print(latex(s))
        print(s)
        x0,y0 = x[0],y[0]
        m1 = -2*a**2*k*l*x0 + 2*a**2*k*y0 - 2*b**2*x0
        m1 = m1/(a**2*l) - (y[0] - k*x[0])
        m1 = m1.expand().simplify()
        # print(latex(m1))

        m1 = m1*(y[0] - k*x[0])*a*a*l
        m1 = m1.expand()
        # print(latex(m1))   

        alpha,beta = symbols("alpha beta")
        s = (xy12 + 2*x[0]*y[0])*alpha
        s = s - alpha*(x[0]*y1y2 + y[0]*x1x2)
        s = s + beta*(y12 - y[0]*y1y2 + y[0]**2)
        s = s - l*(x12 - x[0]*x1x2 + x[0]**2 )  
        # s = s.subs(m,y[0] - k*x[0])
        s = self.simEqn(s,m) 
        print(latex(s)) 

        print(s) 

        m1 = 2*a**2*alpha*k*y0 - 2*a**2*k*l*x0
        m1 = m1 - 2*alpha*b**2*x0 - 2*b**2*beta*y0
        m1 = -m1/(-a**2*l + b**2*beta) - (y[0] - k*x[0])
        m1 = m1.expand().simplify()
        m1 = m1.collect(k)
        print(latex(m1))


        return

    def simEqn(self,s,m):
        """
        docstring for simEqn
        simplify the equation
        """
        s = s.expand().simplify()
        s,denom = fraction(s)
        s = s.expand().collect(m)
        return s


    def fixedPointParabola(self):
        """
        docstring for fixedPointParabola
        """
        p,k,m,l  = symbols("p k m,l")
        alpha,beta = symbols("alpha beta")
        x = symbols("x0:3")
        y = symbols("y0:3")

        x1x2  = -(2*k*m - 2*p)/(k*k)
        x12 = (m/k)**2
        y12  = k*k*x12 + k*m*x1x2 + m*m 
        y1y2 = k*x1x2 + 2*m 
        xy12 = 2*k*x12 + m*x1x2

        s = (xy12 + 2*x[0]*y[0])*alpha
        s = s - alpha*(x[0]*y1y2 + y[0]*x1x2)
        s = s + beta*(y12 - y[0]*y1y2 + y[0]**2)
        s = s - l*(x12 - x[0]*x1x2 + x[0]**2 )  
        # s = s.subs(m,y[0] - k*x[0])
        s = -self.simEqn(s,m) 
        print(latex(s)) 

        print(s)

        x0,y0 = x[0],y[0]
        m1 = 2*alpha*k*y0 + 2*alpha*p 
        m1 = m1 + 2*beta*k*p - 2*k*l*x0
        m1 = m1/l - (y[0] - k*x[0])
        m1 = m1.expand().simplify()
        m1 = m1.collect(k)
        print(latex(m1))

        return

    def getLine(self,coor1,coor2):
        """
        get line equation of two points
        """
        # a*x + b*y + c = 0
        a = coor2[1] - coor1[1]
        b = coor1[0] - coor2[0]
        c = -(a*coor1[0] + b*coor1[1])
        return [a,b,c]

    def getIntersectPoint(self,line1,line2):
        """
        get the intersect point of two lines
        """
        A = np.ones((2,2))
        A[0,:] = line1[:2]
        A[1,:] = line2[:2]
        b = [-line1[2],-line2[2]]
        x = np.linalg.solve(A,b)
        return x
    
    def solveQuadratic(self,arr):
        """
        docstring for solveQuadratic
        arr:
            1d array, [a,b,c]
        """
        a,b,c = arr 
        delta = (b*b - 4*a*c)**0.5 
        x1 = (-b + delta)/(2*a)
        x2 = (-b - delta)/(2*a)

        return [x1,x2]
    def ellipseIntersect(self,line,ellipse):
        """
        docstring for ellipseIntersect
        line:
            [A,B,C]
        ellipse:
            [a,b],(x/a)^2 + (y/b)^2 = 1
        """
        A,B,C = line 
        a,b = ellipse
        arr = []
        arr.append(A**2*a**2 + B**2*b**2)
        arr.append(2*A*C*a**2)
        arr.append(- B**2*a**2*b**2 + C**2*a**2)
        sol = self.solveQuadratic(arr)
        
        res = []
        for x in sol:
            y = (-C-A*x)/B
            res.append([x,y])

        return res

    def fixedOpposite(self):
        """
        docstring for fixedOpposite
        """
        line1 = [1/2,1/3,-1]
        line2 = [1/3,1/2,-1]
        p = self.getIntersectPoint(line1,line2)
        print(p)
        A,B,C,a,b = symbols("A B C a b")
        x,y = symbols("x y")
        s = (x/a)**2 + (y/b)**2 - 1 
        s = s.subs(y,(-C-A*x)/B)
        s = s*(a*B*b)**2
        s = s.expand().collect(x)
        print(s)

        p = [-1,0]
        ellipse = [2,3]

        lines = []
        for k in range(1,10):
            line = [k,-1,p[1] - k*p[0]]
            res  = self.ellipseIntersect(line, ellipse)
            p1 = [res[0][0],-res[0][1]]
            line = self.getLine(p1,res[1])
            lines.append(line)
        for i in range(9):
            for j in range(9):
                if i == j:
                    continue

                p1 = self.getIntersectPoint(lines[i],lines[j])
                print(p1)

        return   

    def pointLinePoint(self):
        """
        docstring for pointLinePoint
        """

        a,b,k,m,l = symbols("a b k m,l")
        x = list(symbols("x0:3"))
        y = list(symbols("y0:3"))

        k = (y[1] - y[0])/x[1]
        m = y[0]
        x1x2 = -2*a*a*k*m/((a*k)**2+b*b)
        x[2] = x1x2 - x[1]
        y[2] = k*x[2] + m 
        s    = y[1] + x[1]*(y[2]-y[1])/x1x2
        s    = s.expand().simplify()
        # print(latex(y))

        x1x2 = x[1]*(y[2]-y[1])/x1x2
        x1x2 = x1x2.expand().simplify()
        print(latex(x1x2))

        
        return


    def fixedPointGeneral(self):
        """
        docstring for fixedPointGeneral
        """

        a,b,c,d,e  = symbols("a b c d e")
        f,g,k,m,l  = symbols("f g k m l")
        alpha,beta = symbols("alpha beta")
        x,y = symbols("x y")
        s = a*x*x+b*x*y+c*y*y+d*x+e*y+f
        s = s.subs(y,k*x+m).expand().collect(x)
        # print(latex(s))


        x = symbols("x0:3")
        y = symbols("y0:3")

        x1x2  = -(b*m+2*c*k*m+d+e*k)/(a+b*k+c*k*k)
        x12 = (c*m*m+e*m+f)/(a+b*k+c*k*k)
        y12  = k*k*x12 + k*m*x1x2 + m*m 
        y1y2 = k*x1x2 + 2*m 
        xy12 = 2*k*x12 + m*x1x2

        s = (xy12 + 2*x[0]*y[0])*alpha
        s = s - alpha*(x[0]*y1y2 + y[0]*x1x2)
        s = s + beta*(y12 - y[0]*y1y2 + y[0]**2)
        s = s - l*(x12 - x[0]*x1x2 + x[0]**2 )  
        # s = s.subs(m,y[0] - k*x[0])
        s = self.simEqn(s,m) 
        print(latex(s)) 
        print("-----------------------")
        print("-----------------------")
        print(s)

        x0,y0 = x[0],y[0]
        m1 = -2*a*alpha*x0 - 2*a*beta*y0 - alpha*b*k*x0 
        m1 = m1 + alpha*b*y0 + 2*alpha*c*k*y0 - alpha*d 
        m1 = m1 + alpha*e*k - b*beta*k*y0 - b*l*x0 
        m1 = m1 - beta*d*k - 2*c*k*l*x0 - e*l
        m1 = -m1/(a*beta - alpha*b - c*l) - (y[0] - k*x[0])
        m1 = m1.expand().simplify()
        m1 = m1.collect(k)
        print(latex(m1))
        
        return


    def ellipseTran(self):
        """
        docstring for ellipseTran
        """
        a,b = symbols("a b")
        b2 = b*b 
        a2 = a*a 
        a6 = a**6
        b6 = b**6
        A = 4*b2+9*a2 
        B = -4*(a2+3*b2)
        C = (a6+4*b6)/(a2*b2)

        delta = (B*B - 4*A*C)
        delta = delta.expand().factor()
        print(latex(delta))

        x,y,x0,y0    = symbols("x y x0 y0")
        alpha,beta,l = symbols("alpha beta l")
        d = a2*l-b2*beta
        e = a2*l+b2*beta
        f = -2*a2*alpha

        d,e,f = symbols("d e f")
        eqn = []
        eqn.append(e*x0+f*y0-d*x)
        eqn.append(f*x0-e*y0-d*y)
        sol = solve(eqn,[x0,y0])

        print(latex(sol[x0]))
        print("-------------------------")
        print(latex(sol[y0]))
        print("-------------------------")
        x0 = sol[x0]
        y0 = sol[y0]
        s = a2*y0**2 + b2*x0**2 - a2*b2 
        s = s.expand().simplify().collect([x,y])
        s,denom = fraction(s)
        print(latex(s))


        f2 = f*f 
        e2 = e*e 
        A = a2*f2+b2*e2 
        B = -2*e*f*(a2 - b2)
        C = a2*e2+b2*f2
        delta = (B*B - 4*A*C)
        delta = delta.expand().factor()
        print("-------------------------")
        print(latex(delta))

        return
    def test(self):
        """
        docstring for test
        """
        # self.abSeven()
        # self.abcDelta()
        # self.nSquare()
        # self.fixedPointConic()
        # self.fixedPointParabola()
        # self.fixedOpposite()
        # self.pointLinePoint()
        # self.fixedPointGeneral()
        self.ellipseTran()

        return

puzzle = Puzzles()
puzzle.test()
        
