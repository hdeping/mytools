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
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from Formulas import Formulas
from Algorithms import Algorithms
from decimal import *
from tqdm import tqdm
import mpmath as mp
import math
from scipy import integrate as inte

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class Puzzles(Algorithms):
    """
    solutions for math puzzles
    """
    def __init__(self):
        super(Puzzles, self).__init__()

        self.gamma = 0.57721566490153286

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
    def solveQuadraticSymbolic(self,arr):
        """
        docstring for solveQuadratic
        arr:
            1d array, [a,b,c]
        """
        a,b,c = arr 
        delta = sqrt(b*b - 4*a*c) 
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
        trajectory of the fixed points
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

    def transformGeneralConic(self,arr):
        """
        arr:
            [a,b,c,d,e,f]
            ax^{2}+bxy+cy^{2}+dx+ey+f=0
            2x/(1-x^2) = b/(a-c)=tan(2theta)
            x = 
        """
        a,b,c,d,e,f = arr
        delta = b*b - 4*a*c
        if delta < 0:
            print("it is an ellipse")
        elif delta > 0:
            print("it is a hyperbola")
        else:
            print("it is a parabola")
            
        if a == c:
            k1 = 1 
            k2 = 1
        else:
            arr = [b,2*(a-c),-b]
            tantheta = self.solveQuadraticSymbolic(arr)[0]
            k1,k2 = fraction(tantheta)
        # sintheta = k1/sqrt(k1*k1+k2*k2)
        # costheta = k2/sqrt(k1*k1+k2*k2)
        costheta = k2 
        sintheta = k1
        x,y,u,v = symbols("x y u v")
        x = costheta*u - sintheta*v
        y = sintheta*u + costheta*v
        s = a*x*x+b*x*y+c*y*y+d*x+e*y+f
        s = s.expand().simplify()
        s,denom = fraction(s)
        s = s.collect([u,v])
        # print(latex(s))
        poly = Poly(s,[u,v]).as_dict()
        print(poly)
        sol = solve([diff(s,u),diff(s,v)],[u,v])
        uv_center = [sol[u],sol[v]]
        s = s.subs(u,sol[u])
        s = s.subs(v,sol[v])
        s = s.expand()
        au = poly[(2,0)]
        bu = poly[(1,0)]
        av = poly[(0,2)]
        bv = poly[(0,1)]
        k  = bu*bu/au/4 + bv*bv/av/4 - poly[(0,0)]
        k  = k.expand().simplify()
        print(k)

        center = self.uv2XY(uv_center,k1,k2)
        print(center)


        return

    def uv2XY(self,uv,k1,k2):

        u,v = uv
        delta = sqrt(k1*k1+k2*k2)
        x = (k2*u - k1*v)/delta
        y = (k1*u + k2*v)/delta

        x = x.expand().simplify()
        y = y.expand().simplify()
        print(latex(x))
        print(latex(y))

        return [x,y] 

    def solveCubic(self,arr):
        """
        docstring for solveCubic
        arr:
            [a,b,c,d]
        """
        a,b,c,d = arr
        q = -(2*b**3-9*a*b*c+27*a**2*d)/2
        p = 3*a*c - b**2 
        delta = (q**2+p**3)**0.5
        m = (q+delta)**(1/3)
        n = (q-delta)**(1/3)
        if delta.imag == 0:
            if q+delta < 0:
                m = -(-q-delta)**(1/3)
            if q-delta < 0:
                n = -(-q+delta)**(1/3)
            
        # print(q,p,delta,m)
        omega1 = -1/2+3**0.5/2*1j
        omega2 = -1/2-3**0.5/2*1j
        x1 = (-b + m + n)/(3*a)
        x2 = (-b + omega1*m + omega2*n)/(3*a)
        x3 = (-b + omega2*m + omega1*n)/(3*a)
        # print(x1+x2+x3)
        # print(x1*(x2+x3)+x2*x3)
        # print(x1*x2*x3)
        # print("-------")
        return [x1,x2,x3]

    def solveQuartic(self,arr):
        """
        docstring for solveQuartic
        arr:
            [a,b,c,d,e]
        """
        a,b,c,d,e = arr 
        A = 8*a*c - 3*b*b 
        C = b**3-4*a*b*c+8*d*a*a
        B = (b*b-4*a*c)**2 + 2*b*C - 64*e*a**3
        arr = [1,A,B,C*C]
        y1,y2,y3 = self.solveCubic(arr)
        print(arr,"C = ",C)
        print("y1 = ",y1)
        print("y2 = ",y2)
        print("y3 = ",y3)
        self.checkRoot([y1,y2,y3])

        y1 = y1**0.5
        y2 = y2**0.5
        y3 = y3**0.5
        print(y1,y2,y3)
        C = y1*y2*y3 
        if C.imag is not 0:
            y1 = -y1*1j
        print("y1y2y3 = ",y1*y2*y3)
        x1 = (-b-y1-y2-y3)/(4*a)
        x2 = (-b+y1+y2-y3)/(4*a)
        x3 = (-b+y1-y2+y3)/(4*a)
        x4 = (-b-y1+y2+y3)/(4*a)


        return [x1,x2,x3,x4]
    def checkRoot(self,roots):
        """
        docstring for checkRoot
        """
        num = len(roots)
        for i in range(1,num+1):
            res = 0 
            combinations = itertools.combinations(roots,i)
            for line in combinations:
                res += np.prod(line)
            print(i,res)
        return
    def distanceEllipse(self):
        """
        docstring for distanceEllipse
        """
        x,y = symbols("x y")
        eqn = [2*sqrt(2)/x-1/y-1]
        s = x*x/2+y*y-1
        eqn.append(s)
        # sol = solve(eqn,[x,y])
        # print(latex(sol[0]))
        # print(len(sol))
        s = s.subs(y,1/(2*sqrt(2)/x-1))
        s = s.expand().simplify()
        s,denom = fraction(s)
        s = s.expand()*2
        print(latex(s))

        return

    def testSolve(self):
        """
        docstring for testSolve
        """
        sol = self.solveCubic([1,1,1,1])
        self.checkRoot(sol)
        arr = [1,-10,35,-50,24]
        arr = [1,-4,4,4,-4]
        sol = self.solveQuartic(arr)
        for x in sol:
            print("-----",x)
        self.checkRoot(sol)
       
        return

    def quarticPlot(self):
        """
        docstring for quarticPlot
        """
        x = Symbol("x")
        s = ((x-1)*(x-2)*(x-3)*(x-4)).expand()
        print(s)
        s = x*(x*(x*(x-4)+4)+4)-4
        s = s.expand()
        print(s.factor())
        print(s.subs(x,-0.946965).expand())
        x = np.linspace(-1.2,1.2,100)
        x = np.linspace(-0.947,-0.946,100)
        # x = np.linspace(0.7747,0.7748,100)
        y = x**4 - 4*x**3 + 4*x**2 + 4*x - 4
        plt.plot(x,y,lw=4)
        plt.plot(x,x-x,lw=4)
        # plt.show()
        return

    def xLogX(self):
        """
        docstring for xLogX
        """
        x = np.linspace(0.001,0.999,100)
        x = np.linspace(1.1,10,100)
        y = x/np.log(x)
        plt.plot(x,y,lw=4)
        plt.plot(x,x-x,lw=4)
        plt.show()
        return

    def nSquare(self):
        """
        docstring for nSquare
        prove that n is a square if 
        n = (a*a+b*b)/(1+ab) is an integer 
        for a,b are integers
        """

        n = 10
        arr = np.arange(1,n)
        combinations = itertools.combinations(arr,2)
        for line in combinations:
            a,b = line
            A = a*a+b*b 
            B = 1+a*b 
            N = A//B 
            if A == N*B:
                print("(%d,%d),"%(a,b))
        n = 300000
        m = 9
        for a in range(2,n):
            A = a**3 - m 
            B = a*m+1
            b = A//B 
            if A == b*B:
                print("((a,b) = %d,%d),"%(a,b))

        a,b,n,t = symbols("a b n t")
        s = (a*a+b*b)*a - (1+a*b)*(b+n)
        s = solve([s],[b])
        b = s[b]
        print(b)
        b = b.subs(a,(t-1)/n)
        b = b.expand()
        # print(latex(b))
        b = b*n**3*t
        b = b.expand()
        print(b)

        a = 8 
        n = 2
        b = 30
        t = a*n + 1
        print(a*a+b*b,a,b,t)
        print((a**3-n)/t)
        print((t-1)**3-n**4)

        k = Symbol("k")
        a = k**5 - k 
        b = k**3 
        s = a*a+b*b 
        s = s.expand()
        print(s)
        s = 1+a*b 
        s = s.expand()
        print(s)


        
        return

    def getDigitsSum(self,a):
        """
        docstring for getDigitsSum
        """
        a = str(a)
        res = 0 
        for i in a:
            res += int(i)

        return res 
    def digitsProblem(self):
        """
        docstring for digitsProblem
        """
        a = 8888**8888 
        for i in range(10):
            a = self.getDigitsSum(a)
            print(i,a)
        return


    def sequenceConverge(self):
        """
        docstring for sequenceConverge
        """
        res = [Integer(2)]
        res = [2]
        for i in range(400000):
            a = res[-1] - 1/res[-1]
            res.append(a)
            # print(i,a)
            if abs(a) < 0.1:
                print(i+1,a)

        x = np.arange(len(res))
        res = np.array(res)

        plt.plot(x,res)
        plt.show()
        return
    def sylowSimpleTest(self,order,factors = None):
        """
        docstring for sylowSimpleTest
        order:
            group order
        according to the Sylow's Theorem
        """
        # print("group order is ",order)

        totalNum = 1
        isNonSimple = False
        if factors == None:
            factors = factorint(order)
        for key in factors:
            index = factors[key]
            if factors == None:
                value = order//(key**index)
                allFactors = self.getAllFactors(value)
            else:
                res = {}
                for i in factors:
                    if i != key:
                        res[i] = factors[i]
                allFactors = self.getAllFactors(1,factors=res)
            res = []
            for i in allFactors:
                if i%key == 1:
                    res.append(i)
            # print("n_{%d} & = & "%(key),res)
            if len(res) > 1:
                totalNum += res[1]*(key-1)
            else:
                totalNum += (key-1)
                isNonSimple = True
                # print("group %d is not simple"%(order))

        return totalNum,isNonSimple

    def testSylow(self):
        """
        docstring for testSylow
        """
        orders = [12,60,6545,1365,2907,
                  132,462,7920,39*97*101*9783]
        for order in orders:
            res,i = self.sylowSimpleTest(order)
            print("totalNum = ",res)

        # factors = { 2:41,3:13,5:6,7:2,
        #             11:1,13:1,17:1,19:1,
        #             23:1,31:1,47:1}

        # print(factors)
        # res = self.sylowSimpleTest(1,factors = factors)
        # print("totalNum = ",res)

        count = 0 
        for i in range(4,10000):
            if isprime(i):
                continue
            # print(i)
            res,judge = self.sylowSimpleTest(i)
            if judge == False and res <= i:
                print("order = ",i,"totalNum = ",res)
                count += 1 
        print(count)

        return

    def IMO1977(self,m,n):
        """
        docstring for IMO1977
        any sum of continuous m terms are negative, 
        any sum of continuous n terms are positive, 
        what is longest length of the sequence?
        """
        if m > n:
            print("%d,%d is invalid, try %d,%d"%(m,n,n,m))
            return
        indeces = [0]
        for i in range(m+n-1):
            if indeces[-1] >= m:
                num = indeces[-1] - m 
                indeces.append(num)
            else:
                num = indeces[-1] + n 
                indeces.append(num)

        print(indeces)

        Sn = np.arange(1,m+n)
        for j,i in enumerate(indeces[1:]):
            i = i-1
            Sn[i] = j+1
        print(Sn)
        Sn = Sn[1:] - Sn[0:-1]
        print(Sn)
        for k2 in [m,n]:
            k1 = len(Sn)-k2
            for i in range(k1):  
                print(i,sum(Sn[i:i+k2]))

        print(len(Sn))

        res = 0 
        for j,i in enumerate(Sn):
            res = res + i
            print(j,res)
        
        matrix = np.ones((m-1,n),int)
        for i in range(m-1):
            matrix[i,:] = Sn[i:i+n]
        print(matrix)
        print(np.sum(matrix,axis=0))
        print(np.sum(matrix,axis=1))
        return

    def getSigmaN(self,alpha,n=20):
        """
        docstring for getSigmaN
        \sigma_{\alpha}(n)&=&\sum_{d\mid n}d^{\alpha}
        """
        res = [0,1]
        for i in range(2,n+1):
            factors = self.getAllFactors(i)
            num = 0 
            for j in factors:
                num += j**alpha 
            res.append(num)
        
        return res

    def getEllipticDelta(self):
        """
        docstring for getEllipticDelta
        """
        n = 400
        A = self.getSigmaN(3,n=n)
        B = self.getSigmaN(5,n=n)
        A2 = self.polynomialPow(A,2)[:n+1]
        A3 = self.polynomialPow(A,3)[:n+1]
        B2 = self.polynomialPow(B,2)[:n+1]
        # print(A)
        # print(B)
        # print(A2)
        # print(A3)
        # print(B2)
        Delta = []
        for i in range(n+1):
            num = (5*A[i]+7*B[i])//12
            num += 100*A2[i] - 147*B2[i] + 8000*A3[i]
            # print(num)
            Delta.append(num)

        g2 = A.copy()
        for i in range(len(g2)):
            g2[i] *= 240 
        g2[0] = 1 
        # print(g2)
        g23 = self.polynomialPow(g2,3)[:n+1]
        # print(g23)

        Delta = Delta[1:]
        # print(Delta)


        jTau = self.polynomialDivide(g23,Delta)
        # print(jTau[:100])
        for i in range(21):
            cn = self.asymCn(i)
            print("c(%d) & = & %d \\\\"%(i,jTau[i+1]),cn)

        n = 385
        c385 = jTau[n+1]
        a1 = (c385 // 25)
        a2 = (c385 // 7)
        a3 = (c385 // 11)
        print("c%d = %d"%(n,c385))
        print(a1*25)
        print(a2*7)
        print(a3*11)
        
        print(self.asymCn(n))

        for i,j in enumerate(Delta):
            print(i+1,j)


        return

    def asymCn(self,n):
        """
        docstring for asymCn
        """
        if n == 0:
            return 1
        cn = np.exp(4*np.pi*n**0.5)
        cn = cn/(2**0.5*n**(3/4))
        return cn


    def getQuarticArea(self):
        """
        docstring for getQuarticArea
        S(x^4+y^4 = x^2+y^2)
        """
        u,v = symbols("u v")
        x,y = symbols("x y")

        s = x**4+y**4-x**2-y**2 
        s = s.subs(x,u+v)
        s = s.subs(y,u-v)
        s = s.expand()

        print(s)
        return

    def checkDiophantine(self):
        """
        docstring for checkDiophantine
        check the equation
        20D^{2020}&=&49A^{1949}+79B^{1979}+19C^{2019}
        """
        a = 2
        b = 2
        m = 50000
        base = 13
        n = base**m 
        phi = (base - 1)*n//base
        X = [30020,37240,77420,73549,b,a,a+b]
        res = []
        for i in X:
            res.append(i%n)
        X = res 
        print("X = ",X)
        exponents = [[2687610569,6427526880,5704546660,
                      1322543931,23356556334698066540,
                      25901953851181467018439618,
                      26135095002607441931954713422],
                     [2646868620,6330090899,5618070460,
                      1302495261,23002490296274144359,
                      25509301695782051146507739,
                      25738908620556798547437966882],
                     [2594429420,6204680480,5506766439,
                      1276690501,22546769834733299498,
                      25003916818203407240682920,
                      25228974819258001151748259762],
                     [2593145049,6201608856,5504040317,
                      1276058476,22535608067488382023,
                      24991538641560732286603374,
                      25216485227763318972960265574]]
        res = []
        for line in exponents:
            tmp = []
            for i in line:
                tmp.append(i%phi)
            res.append(tmp)

        exponents = res

        # check A, B, C and D

        # 20D^{2020}&=&
        # 49A^{1949}+79B^{1979}+19C^{2019}

        arr = []

        indeces = [1949,1979,2019,2020]
        coef = [49,79,19,20]

        final = []
        for i,line in enumerate(exponents):
            print(i)
            res = 1 

            for x,y in zip(X,line):
                # print(x,y)
                res *= self.getModN(x,y,n)
                res  = res%n

            res = self.getModN(res,indeces[i],n)
            res = (coef[i]*res)%n
            final.append(res)

        print(sum(final[:3])%n - final[-1])


        return

    def getGamma(self,z,n=100000):
        """
        docstring for getGamma
        -\ln\Gamma(z) &=& \ln z+\gamma z+\sum_{k=1}^{\infty}
        \left[\ln\left(1+\frac{z}{k}\right)-\frac{z}{k}\right]

        """
        res = np.log(z) + self.gamma*z
        arr = np.arange(1,n+1)
        arr = z/arr
        arr = np.log(1+arr) - arr 
        res = res + sum(arr)
        res = np.exp(-res)

        return res 

    def testGamma(self):
        """
        docstring for testGamma
        """
        z = 0.5
        res = self.getGamma(z) 
        print(z,res,res**2)
        print(self.getGamma(1.46163))
        res = self.getGamma(1/4)
        print(res,res**2/2/np.pi**0.5)

        res = self.getGamma(1j,n=1000000)
        print(res)
        res = self.getGamma(1j,n=100000)
        print(res)
        # arr = np.linspace(0.01,6,100)
        # y = []
        # for x in arr:
        #     y.append(self.getGamma(x))
        # plt.plot(arr,y,lw=4)
        # plt.savefig("gamma.png",dpi=300)
        # plt.show()

        res = []
        N = 100
        delta = 0.1
        for i in tqdm(range(1,N)):
            x = 0.9 + i*delta*1j
            y = self.getGamma(x)
            res.append(abs(y))
        plt.plot(np.arange(1,N)*delta,res,lw=4)
        # plt.savefig("gamma.png",dpi=300)
        plt.show()


        return

    def getExp163(self,n=163,nine=0,epi=1,exponent=2):
        """
        docstring for getExp163
        """

        getcontext().prec = 100
        a = Decimal(n)
        if nine == 1:
            a = a/Decimal(9)
        a = a**(Decimal(1)/Decimal(exponent))
        pi = Decimal(31415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421)
        # print("      ",pi)
        pi = pi/Decimal(10**94)
        # print("pi = ",pi)
        e = Decimal(271828182845904523536028747135266249775724709369995957496696762772407663035354759457138217852516)
        # print("e =  ", e)
        e = e/Decimal(10**95)
        # print("e = ", e)
        # print("a = ",a,a**Decimal(2))
        if epi == 0:
            n = pi**(e*a)
        else:
            n = e**(pi*a)
        # print(n)
        return n

    def testExp163(self):
        """
        docstring for testExp163
        """
        # print(self.getExp163(n=1))
        # for i in tqdm(range(1,10)):
        for j in range(3,10):
            print("j = ",j)
            for i in tqdm(range(1,30000)):
                res = self.getExp163(n=i,epi=0,exponent=j)
                res = str(res).split(".")
                digits = res[1]
                # print("%5d ==> "%(i),res)
                m = 5
                if digits[:m] == "9"*m:
                    print(i,res[0],res[1][:10])

        res = self.getExp163(nine=1)
        a = Decimal(1)/(res - Decimal(640320))
        print(a,int(a))
        for i in range(1):
            a = Decimal(1)/(a - Decimal(int(a)))
            print(i,int(a))
        return

    def getMatrixInverse(self):
        """
        docstring for getMatrixInverse
        a 20000X20000 matrix is zero everywhere 
        except 2,3,5,...,224737 on the diagonal
        and a_{ij} = 1 when abs(i-j)=2**m
        what is A^{-1}_{11}?
        """

        n = 20
        mat = Matrix.zeros(n)
        # print(mat)

        arr = []
        for i in range(n):
            arr.append(1<<i)
            mat[i,i] = prime(i+1)
            # mat[i,i] = 1
        # print(arr)
        for i in range(n):
            for j in range(i+1,n):
                res = j-i
                # print(i,j,res)
                if res in arr:
                    mat[i,j] = 1
                    mat[j,i] = 1
        # print(mat)
        # print(mat.inv())
        a = mat.det()
        b = mat[1:,1:].det()
        print(b,a,b/a)

        # print(latex(mat))

        
        return

    def tetraPolyhedron(self):
        """
        docstring for tetraPolyhedron
        """
        a,b,c,a1,b1,c1 = symbols("a b c a1 b1 c1")

        cosA = (b**2+c**2-a1**2)/(2*b*c)
        cosB = (b**2+a**2-c1**2)/(2*b*a)
        cosC = (a**2+c**2-b1**2)/(2*a*c)

        V = 1-cosA**2-cosB**2-cosC**2+2*cosA*cosB*cosC
        V = V*4*(a*b*c)**2 
        V = V.expand()
        print(latex(V))


        # s = a**4+4*a**3+4*a**2+1
        # print(s.factor())
        # res = self.solveQuartic([1,4,4,0,1])
        # print(res)
        # self.checkRoot(res)
        return


    def getTanArctan(self,m):
        """
        docstring for tanTanArctan
        \arctan\frac{1}{m}&=&\arctan\frac{1}{x}+\arctan\frac{1}{y}
        """
        n = m*m + 1 
        factors = self.getAllFactors(n)
        length  = (len(factors)+1)//2 
        # print(factors,length)
        res = []
        for a in factors[:length]:
            b = n // a 
            line = [m + a,m+b]
            res.append(line)

        return res
    def arctanAddTwo(self,a,b):
        """
        docstring for arctanAddTwo
        tan(arctan a + arctan b)
        """
        res = (a+b)/(1-a*b)
        return res
    def arctanTimes(self,a,n):
        """
        docstring for arctanTimes
        tan(n*arctan a)
        """
        res = a 
        if n == 1:
            return res
        for i in range(abs(n)-1):
            res = self.arctanAddTwo(res,a)
        if n < 0:
            res = - res

        return res 
    def arctanAddArray(self,arr):
        """
        docstring for arctanAddArray
        """
        length = len(arr)
        res = arr[0]
        for i in range(1,length):
            res = self.arctanAddTwo(res,arr[i])

        return res

    def arctanIntArray(self,arr):
        """
        docstring for arctanDicts
        arr:
            2d array, [n,m]
        return:
            tan(\sum n*arctan (1/m))
        """
        res = []
        for line in arr:
            n,m = line 
            num = 1/Integer(m) 
            num = self.arctanTimes(num,n)
            res.append(num)
        res = self.arctanAddArray(res)

        return res 

    def testTanArctan(self):
        """
        docstring for testTanArctan
        """
        types = "\\arctan\\frac{1}{%d}"
        string = "%s &=& %s + %s"%(types,types,types)
        for m in range(2,3):
            res = self.getTanArctan(m)
            for line in res:
                print(string%(m,*line))

        a = self.arctanTimes(1/Integer(18),12)
        b = self.arctanTimes(1/Integer(57),8)
        c = self.arctanTimes(1/Integer(239),-5)
        print(a,b,c)
        print(self.arctanAddArray([a,b,c]))
        arr = [[12,18],[8,57],[-5,239]]
        arr = [[2,2],[3,3]]
        arr = [[2,70],[-2,12],[1,7]]
        arr = [[12,5],[-4,7],[-8,8]]
        print(self.arctanIntArray(arr))
        arr = [[12,18],[8,57]]
        print(self.arctanIntArray(arr))
        arr = [[4,5],[-1,239]]
        print(self.arctanIntArray(arr))
        arr = [[5,5],[-3,18],[-2,57]]
        print(self.arctanIntArray(arr))



        return

    def testArctanTimes(self):
        """
        docstring for testArctanTimes
        """
        n,m = symbols("n m")
        for order in tqdm(range(2,20)):
            s = self.arctanTimes(1/n,order)
            s = s.simplify()
            # print(latex(s))
            s1,s2 = fraction(s)
            s = m*s1-s2*(m+1)
            for k in range(2,10):
                # print(k,s)
                S = s.subs(n,k)
                x = solve(S,[m])[0]
                # print(order,k,x)
                if x.is_integer:
                    print(order,k,x)
        return

    def testZetaNegativeOdd(self):
        """
        docstring for testZetaNegativeOdd
        """
        for i in range(2):
            i = -2*i-1
            res = mp.zeta(i)
            print(i,res)

        a = 0.0210927960927961*2730*12 
        print(a)

        # print(mp.siegel(2))
        X = np.linspace(0,100,500)
        Y = []
        for i in tqdm(X):
            Y.append(mp.siegelz(i))
        plt.plot(X,Y,lw=4)
        plt.plot(X,X-X)
        plt.show()

        return

    def testZeta(self):
        """
        docstring for testZeta
        """
        X = np.linspace(0,100,500)
        Y = []
        for i in tqdm(X):
            Y.append(abs(mp.zeta(i*1j)))
        plt.plot(X,Y,lw=4)
        plt.plot(X,X-X)
        plt.show()

        return
    def showPlot(self,f,begin=0,end=10,N=500):
        """
        docstring for showPlot
        """
        X = np.linspace(begin,end,N)
        Y = []
        for i in tqdm(X):
            Y.append(abs(f(i)))
        plt.plot(X,Y,lw=4)
        plt.plot(X,X-X)
        plt.show()
        return
    def testKleinj(self):
        """
        docstring for testKleinj
        """
        print('plot begins')
        
        mp.cplot(lambda t: mp.kleinj(tau=t), 
            [0.3,0.35], [0,0.1], 
            points=50000,
            file="kleinj2.png",
            dpi=500)
        print('plot done!')

        return

    def testEllipk(self):
        """
        docstring for testEllipk
        """
        print('plot begins')
        
        # mp.cplot(lambda t: mp.ellipk(t), 
        #     [0,10], [0,10], 
        #     points=10000,
        #     file="ellipk.png",
        #     dpi=500)
        # print('plot done!')

        self.showPlot(mp.ellipk,
                      begin=-1000,end=1000,
                      N = 10000)

        return

    def testSquareEqn(self):
        """
        docstring for testSquareEqn
        """
        self.square = 1
        N = 5000
        n = int((N+0.01)**0.5)
        # res = self.getCombinatorEqnRecursive(n,N)
        # print(N,len(res),res)

        res = self.getCombinatorEqnSolNumByIter(n,N,square=True)
        res = np.log(res[1:])
        X   = np.arange(len(res)) + 1 
        X   = np.sqrt(X)
        # X   = np.log()
        n = 2000
        X = X[-n:]
        res = res[-n:]
        corr = np.corrcoef(X,res)
        print(corr)
        print(np.polyfit(X,res,1))
        plt.plot(X,res,lw=2)
        plt.plot(X,0.51795424*X + 7.15312448,lw=2)
        plt.show()
        # print(res)
        return

    def testDedekindEta(self):
        """
        docstring for testDedekindEta
        """
        res = [1,-1]
        for i in range(2,30):
            arr     = [0]*(i+1)
            arr[0]  = 1 
            arr[-1] = -1
            # print(i,arr)
            res = self.polynomialMulti(res,arr)
        print(res)
        res = self.polynomialPow(res,24)
        print(res[:20])
        return
    def testBefore(self):
        """
        docstring for testBefore
        """
        # self.abSeven()
        # self.abcDelta()
        # self.nSquare()
        # self.fixedPointConic()
        # self.fixedPointParabola()
        # self.fixedOpposite()
        # self.pointLinePoint()
        # self.fixedPointGeneral()
        # self.ellipseTran()
        # arr = [7,24,3489,4,5,6]
        # self.transformGeneralConic(arr)
        # self.distanceEllipse()
        # self.testSolve()
        # self.quarticPlot()
        # self.xLogX()
        # self.nSquare()
        # self.digitsProblem()
        # self.sequenceConverge()
        # self.testSylow()
        # self.IMO1977(7,11)
        # self.getEllipticDelta()
        # self.getQuarticArea()
        # self.checkDiophantine()
        # self.testGamma()
        # self.getExp163()
        # self.testExp163()
        # self.getMatrixInverse()
        # self.tetraPolyhedron()
        # self.testTanArctan()
        # self.testZetaNegativeOdd()
        # self.testZeta()
        # self.testKleinj()
        # self.testEllipk()
        # self.testSquareEqn()
        # self.testDedekindEta()

        # self.testRamanujanPi1()
        # self.alibabaPuzzles()
        # self.testGalois()
        # self.testEqnDet()
        # self.testCharacters()
        # self.testFrobenius()
        # self.symmetryGroup()
        # self.solvableQuintic()
        # self.testCharacterExercise()
        # self.testAerodynamics()
        # self.testXX()
        # self.testABElliptic()
        
        # self.getXXTaylor()
        # self.testPseudo()
        # self.testLucasSequence()
        # self.testCoinFountain()

        # self.testFiboFrac()
        # self.testTaylorSeries()
        # self.testLucasSum()
        # self.testIntegral()
        # self.testExpCos()

        # self.testRamaMagicSquare()
        # self.testArctanSqrt()
        # self.testFermatNum()
        # self.testABCConjecture()
        # self.testCollatz()
        # self.tangentPower()
        # self.arctanIntegral()
        # self.getEulerNumbers()
        # self.sphericalPacking()
        # self.thetaSeries()
        # self.testPolytope()
        # self.hexaCode()
        # self.sphericalCrown()
        # self.getGCoef()
        # self.golayCode()
        # self.steinerSystem()
        # self.steiner45n()
        # self.steiner5824()
        # self.origamiCubic()

        # about differential geometry

        # self.curvature()
        # self.surfaceArea()
        # self.spherialTriangle()
        # self.secondForm()
        # self.xyGeodesic()
        # self.christoffel()
        # self.mullerPotential()
        # self.mobiusStrip()
        # self.mullerSurface()
        
        return

    def getPiE(self):
        """
        docstring for getPiE
        """
        getcontext().prec = 100
        pi = Decimal(31415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421)
        # print("      ",pi)
        pi = pi/Decimal(10**94)
        # print("pi = ",pi)
        e = Decimal(271828182845904523536028747135266249775724709369995957496696762772407663035354759457138217852516)
        # print("e =  ", e)
        e = e/Decimal(10**95)
        return pi,e

    def testRamanujanPi1(self):
        """
        docstring for testRamanujanPi1
        """
        pi,e = self.getPiE()
        epsilon = lambda x: np.log(x+(x*x-1)**(0.5))
        X = []
        sq2  = 2**0.5
        sq34 = 34**0.5
        X.append(429    + 304*sq2)
        X.append(627/2  + 221*sq2)
        X.append(1071/2 + 92*sq34)
        X.append(1553/2 + 133*sq34)

        r_pi = np.log(2)
        for i in range(4):
            r_pi = r_pi + epsilon(X[i])
        factor = 6/3502**0.5
        r_pi = r_pi*factor
        print("pi = ",r_pi)

        X = []
        sq2  = Decimal(2**0.5)
        sq34 = Decimal(34**0.5)
        two  = Decimal(2)
        X.append(Decimal(429)+Decimal(304)*sq2)
        X.append(Decimal(627)/two+Decimal(221)*sq2)
        X.append(Decimal(1071)/two+Decimal(92)*sq34)
        X.append(Decimal(1553)/two+Decimal(133)*sq34)
        # print(X)
        res = Decimal(2)
        half = Decimal(0.5)
        epsilon = lambda x: x+(x*x-Decimal(1))**half
        for i in range(4):
            res = res*epsilon(X[i])
        res = res**Decimal(6)
        factor = Decimal(3502)**half
        # factor = factor/Decimal(6)
        print(res)
        print(e**(pi*factor)/res)

        kk = (X[0] - Decimal(429))/Decimal(304)


        return

    def alibabaPuzzles(self):
        """
        docstring for alibabaPuzzles
        """
        y,z = symbols("y z")
        s = y**3 - 3*y*z - (z**3 + 1)
        print(s.factor())
        # answer: y = z + 1 

        # 3*4 tree planting problems
        arr = np.arange(4)
        combinations = itertools.combinations(arr,2)
        combinations2 = itertools.combinations(arr[:3],2)
        lines2 = []
        for line2 in combinations2:
            lines2.append(line2)

        combinations2 = itertools.product(lines2,repeat=2)
        lines2 = []
        for line2 in combinations2:
            lines2.append(line2)
        count = 0
        for line1 in combinations:
            for line2 in lines2:
                output = np.zeros((3,4),int)
                count += 1
                for i in range(2):
                    for j in range(2):
                        output[line2[i][j],line1[i]] = 1

                print(count,output)
                break
        # consider the symmetry
        

        s = Integer(3)/4
        t = Integer(4)/5
        x = Symbol("x")
        n = Symbol("n")

        eqn = (x-s)*(x-t) - (1-s)*(1-t)
        eqn = eqn.factor()
        print(eqn)
        a = Matrix([[s,1-t],[1-s,t]])
        P,D = a.diagonalize()
        P = P/3
        print(P,D**n,P**(-1))
        r = Integer(1)/3
        r = 0
        M = Matrix([[r],[1-r]])

        res = P*(D**n)*(P**(-1))*M 
        print(latex(res))




        return

    def getSnNewtonByArray(self,arr,N=5):
        """
        docstring for getSnNewtonByArray
        arr:
            [a1,a2,a3,...]
        """
       
        M = []
        n = len(arr)
        for i in range(n):
            M.append((i+1)*arr[i])
        M = -Matrix(M)

        A = eye(n)
        for i in range(1,n):
            for j in range(n-i):
                A[i+j,j] = arr[i-1]
       
        A_inv = self.getInverseSymbolMatrix(A)
        S = A_inv*M
        S = S.expand()
        res = [n] + list(S)
        if N <= n:
            return res[:N+1]
        else:
            for i in range(n+1,N+1):
                num = 0 
                for j in range(n):
                    num = num - res[-j-1]*arr[j]
                res.append(num)


        return res

    def getEqnDet(self,arr):
        """
        docstring for getEqnDet
        arr:
            [a1,a2,...]
        """
        n = len(arr)
        A = eye(n)
        res = self.getSnNewtonByArray(arr,N=2*n-2)
        # print(res)
        for i in range(n):
            for j in range(n):
                A[i,j] = res[i+j]
        # print(latex(A))
        D = A.det()
        D = D.expand()
        
        return D

    def testGalois(self):
        """
        docstring for testGalois
        """
        a = symbols("a0:10")
        x = symbols("x0:10")

        s = 1 
        n = 2
        for i in range(1,n+1):
            for j in range(i+1,n+1):
                s = s*(x[i] - x[j])**2
        s = s.expand()
        # print(latex(s))
        print(s)

        arr = symbols("a1:4")
        # arr = [1,Integer(1)/2,Integer(1)/3]

        # res = self.getSnNewtonByArray(arr)
        # print(res)
        p,q,r= symbols("p q r")

        arr = [0]*8 + [p,q,r]
        for i in range(0):
            D = self.getEqnDet(arr)
            # print("D_%d & = & %s \\\\"%(len(arr),latex(D)))
            res = Poly(D).as_dict()
            # print(res)
            line = []
            for key in res:
                line.append(res[key])
            print(len(arr),line[1:-2])
            arr = [0] + arr

        res = []
        res.append(Integer(-4)/ 18)
        res.append(Integer(-27)/ 144)
        res.append(Integer(256)/ -1600)
        res.append(Integer(3125)/ -22500)
        res.append(Integer(-46656)/ 381024)
        res.append(Integer(-823543)/ 7529536)
        res.append(Integer(16777216)/ -169869312)
        res.append(Integer(387420489)/ -4304672100)
        res.append(Integer(-10000000000)/ 121000000000)
        res.append(Integer(-285311670611)/ 3734989142544)
        for j,i in enumerate(res):
            print(j+3,i)

        res = []
        res.append(Integer(144)/ -128)
        res.append(Integer(-1600)/ 2250)
        res.append(Integer(-22500)/ 43200)
        res.append(Integer(381024)/ -926100)
        res.append(Integer(7529536)/ -22127616)
        res.append(Integer(-169869312)/ 585252864)
        res.append(Integer(-4304672100)/ 17006112000)
        res.append(Integer(121000000000)/ -539055000000)
        res.append(Integer(3734989142544)/ -18520607318400)
        for j,i in enumerate(res):
            n = j+4
            print(n,i,Integer(2)*(n-1)**2/n/(n-2)**2)

        res = []
        res.append(Integer(43200)/ -13824)
        res.append(Integer(-926100)/ 600250)
        res.append(Integer(-22127616)/ 21676032)
        res.append(Integer(585252864)/ -768144384)
        res.append(Integer(17006112000)/ -27993600000)
        res.append(Integer(-539055000000)/ 1067328900000)
        res.append(Integer(-18520607318400)/ 42857603712000)

        for j,i in enumerate(res):
            n = j+5
            print(n,i,Integer(3)*n**2/(n+1)/(n-1)/(n-4))

        res = []
        res.append(Integer(21676032)/ -3538944)
        res.append(Integer(-768144384)/ 283553298)
        res.append(Integer(-27993600000)/ 16588800000)
        res.append(Integer(1067328900000)/ -880546342500)
        res.append(Integer(42857603712000)/ -45539366400000)
        res.append(Integer(-1816180145455104)/ 2367182715625728)
        res.append(Integer(-81225792691610112)/ 125603592643436544)
        res.append(Integer(3829976981812800000)/ -6858785309266800000)
        res.append(Integer(190115735040000000000)/ -387144769536000000000)
        res.append(Integer(-9917889551655763968000)/ 22639713698237644800000)
        res.append(Integer(-542786611664628448985088)/ 1373075314682094431502336)
        res.append(Integer(31108603786691939470319616)/ -86392213822962601743301632)
        res.append(Integer(1863922796594312077468800000)/ -5638237379171714095833600000)
        for j,i in enumerate(res):
            n = j+8
            i = -i*n*(n-2)*(n-6)*(n-7)/(n-1)**2/(n-5)/4
            a,b = fraction(i)
            print(n,i,factorint(a),factorint(b))

        # for j,i in enumerate(range(1,len(res))):
        #     n = j+6
        #     num = res[i]/res[i-1]
        #     a,b = fraction(a,b)
        #     print(n,factorint(num))

        res = []
        res.append(Integer(16588800000)/ -1638400000)
        res.append(Integer(-880546342500)/ 209217810978)
        res.append(Integer(-45539366400000)/ 18065203200000)
        res.append(Integer(2367182715625728)/ -1343282255295552)
        res.append(Integer(125603592643436544)/ -93645282089189376)
        res.append(Integer(-6858785309266800000)/ 6368872072890600000)
        res.append(Integer(-387144769536000000000)/ 431675021249740800000)
        res.append(Integer(22639713698237644800000)/ -29521672123554201600000)
        res.append(Integer(1373075314682094431502336)/ -2052486283538632506605568)
        res.append(Integer(-86392213822962601743301632)/ 145750966625349536559330816)
        res.append(Integer(-5638237379171714095833600000)/ 10602652958379099874713600000)

        for j,i in enumerate(res):
            n = j+10
            i = -i*(n-2)*(n)*(n-9)*(n-8)/5/(n-1)**2/(n-6)
            a,b = fraction(i)
            print(n,i,factorint(a),factorint(b))

        res = []
        res.append(Integer(18065203200000)/ -1194393600000)
        res.append(Integer(-1343282255295552)/ 222325651050074)
        res.append(Integer(-93645282089189376)/ 26597476569710592)
        res.append(Integer(6368872072890600000)/ -2640157428175312500)
        res.append(Integer(431675021249740800000)/ -238753591999856640000)
        res.append(Integer(-29521672123554201600000)/ 20584447164275097600000)
        res.append(Integer(-2052486283538632506605568)/ 1735477549591646485610496)
        res.append(Integer(145750966625349536559330816)/ -145301117962925618236616832)
        res.append(Integer(10602652958379099874713600000)/ -12199941609897110446080000000)

        print("m = 6")
        for j,i in enumerate(res):
            n = j+12
            i = -i*(n-2)*(n)*(n-10)*(n-11)/6/(n-1)**2/(n-7)
            a,b = fraction(i)
            print(n,i,factorint(a),factorint(b))



        return

    def getIntegerArray(self,arr):
        """
        docstring for getIntegerArray
        """
        res = []
        for i in arr:
            res.append(Integer(i))
        return res
    def testEqnDet(self):
        """
        docstring for testEqnDet
        """
        p,q = symbols("p q")
        n = 5 
        # p,q = -1
        arr = [0]*(n-2) + [p,q]
        D = self.getEqnDet(arr)
        print(D)

        f = lambda p,q:256*p**5 + 3125*q**4 
        N = 3
        for p in range(-N,N):
            for q in range(N):
                res = f(p,q)
                # print(p,q,res)
                # if res == 0:
                #     print("0: ",p,q,res)
                if sqrt(res).is_integer:
                    print("square: ",p,q,res)

        x = Symbol("x")
        p,q = -5,12
        s = x**5 + p*x + q 
        print(s.factor())
        res = f(p,q)
        print(res,factorint(res))

        arr1 = [1,0,0,0,-5,4]
        arr2 = [1,-1]
        arr1 = self.getIntegerArray(arr1)
        arr2 = self.getIntegerArray(arr2)
        res,remain = self.polynomialFactor(arr1,arr2)
        print(res,remain)

        arr1 = [1,1,1]
        arr2 = [1,0,1,1]
        res = self.polynomialMulti(arr1, arr2)
        print(res)
        arr = [0,-14,0,56,0,-56,22]
        D = self.getEqnDet(arr)
        print(D,factorint(D))
        x = Symbol("x")

        y = self.getPolynomialValues(res,x)
        print(y.factor())
        return

    def getOutZeros(self,arr,n=120):
        """
        docstring for getOutZeros

        get rid of zeros and add number of zeros
        """
        res = []
        for i in arr:
            if i is not 0:
                res.append(i)
        if sum(res) < n:
            res.append(n-sum(res))

        res.sort()

        return res
    def testCharacters(self):
        """
        docstring for testCharacters
        test \chi
        """
        
        # characters for S5, |S5| = 120
        # 7 kinds of conjugacy classes
        sizes = [1,10,20,15,30,20,24] 

        self.square = True
        N = 120
        n = int((N+0.01)**0.5)

        res = self.getCombinatorEqnRecursive(n,N)
        print(len(res))
        for line in res:
            arr = self.getOutZeros(line)
            # print(arr,line)
            if len(arr) == 7:
                print(arr,line)
            # break


        return

    def testFrobenius(self):
        """
        docstring for testFrobenius
        A Frobebius(20) group is defined by
        x,y&:&x^{4}=y^{5}=1
        xyx^{-1}&=&y^{2}
        x^{n}y^{m}x^{-n}&=&y^{2^{n}m\left(mod\;5\right)}
        """
        strings = "x^{%d}y^{%d}&=&y^{%d}x^{%d} \\\\"
        for n in range(1,4):
            for m in range(1,5):
                k = (m*2**n)%(5)
                print(strings%(n,m,k,n))
        X = []
        for n in range(1,4):
            X.append("x%d"%(n))
        Y = []
        for n in range(1,5):
            Y.append("y%d"%(n))

        f20 = ["e"] + X + Y
        for x in X:
            for y in Y:
                f20.append(x+y)
        print(f20)

        
        return 

    def getFactorials(self,n):
        """
        docstring for getFactorials
        n:
            integer
        return:
            [1!,2!,...,n!]
        """
        res = []
        num = 1 
        for i in range(1,n+1):
            num = num*i 
            res.append(num)

        return res

    def getNewForm(self,arr):
        """
        docstring for getNewForm
        [2,2,1,0] => [1,1,2,2,3]
        """
        new_form = []
        for i,j in enumerate(arr):
            if j > 0:
                new_form += [i+1]*j

        return new_form 
    
    def getSignCombination(self,arr,line):
        """
        docstring for getSignCombination

        A = {a:a=+/- 1}, A is a set of 1 and -1 
        sum(arr*line*A) = 0

        try to find a candidate for the characters 
        of a group
        """
        assert(len(arr) == len(line))
        n = len(arr)
        products = itertools.product([1,-1],repeat = n-1)
        res = []
        arr = np.array(arr)
        line = np.array(line)
        for tmp in products:
            out = line[1:]*np.array(tmp)
            out = [line[0]] + out.tolist()
            out = np.array(out)
            if sum(arr*out) == 0:
                out = out.tolist()
                if out not in res:
                    res.append(out)
        
        return res

    def str2List(self,string):
        """
        docstring for str2List
        """
        string = string[1:-1]
        string = string.split(",")
        res = []
        for i in string:
            res.append(int(i))

        return res 
    def getMultiValue(self,coefs,line1,line2):
        """
        docstring for getMultiValue
        """

        coefs = np.array(coefs)
        arr1 = np.array(line1)
        arr2 = np.array(line2)
        res  = sum(coefs*arr1*arr2)

        return res 

    def getOrthogonalChi(self,coefs,rows,conju):
        """
        docstring for getOrthogonalChi
        """
        combi = itertools.combinations(rows,2)
        count = 0

        dicts = {}
        for line in combi:
            res = self.getMultiValue(coefs,*line)
            if res == 0:
                count += 1 
                # print(count,line)
                for key in line:
                    key = str(key)
                    if key in dicts:
                        dicts[key] += 1 
                    else:
                        dicts[key] = 1

        # print(dicts)
        res = []
        for key in dicts:
            val = dicts[key]
            if val >= conju - 1:
                # print(key)
                res.append(self.str2List(key))

        # print(dicts)

        return res,dicts.values()
    
    def checkCharacterTable(self,table,coefs,n):
        """
        docstring for checkCharacterTable
        n:
            order of the group
        coefs:
            1d array, size of the conjugacy classes

        S5:
            n = 120, coefs = [1,10,15,20,20,30,24]
        """
        table = np.array(table)
        coefs = np.array(coefs)

        # check orthoganal
        m = len(table)
        combi = itertools.combinations(np.arange(m),2)
        out = []
        for line in combi:
            i,j = line
            res = sum(coefs*table[i,:]*table[j,:])
            out.append(res)
            if res != 0:
                return 0 
            res = sum(table[:,i]*table[:,j])
            if res != 0:
                return 0 
        # print(out)

        out1,out2 = [],[]
        for i in range(m):
            res1 = sum(coefs*table[i,:]**2)
            out1.append(res1)
            res2 = sum(table[:,i]**2)
            out2.append(res2)
            if res1 != n:
                return 0 
            if res2 != n//coefs[i]:
                return 0 
        # print(out1,out2)
        return 1

    def combination2List(self,arr,m):
        """
        docstring for combination2List
        """
        res = itertools.combinations(arr,2)
        out = []
        for line in res:
            out.append(line)

        return out 
    def symmetryGroup(self):
        """
        docstring for symmetryGroup
        permutation group, S_n
        order: n!
        conjugacy classes: p(n)
        """

        n = 5
        self.square = 0
        res = self.getCombinatorEqnRecursive(n,n)
        factorials = self.getFactorials(n)
        # print(factorials)
        coefs = []
        for line in res:
            num = factorials[n-1]
            new_form = []
            for i,j in enumerate(line):
                if j > 0:
                    tmp = (i+1)**j*factorials[j-1]
                    num = num // tmp
                    if i > 0:
                        new_form += [i+1]*j

            coefs.append(num)

        self.square = 1
        conju = len(res)

        print("coefs",coefs)
        coefs = np.array(coefs)
        N   = factorials[n-1]
        res = self.getSquareEqn(coefs,N)
        count = 0 

        rows = []
        for line in res:
            if line[0] > 0:
                tmp = self.getSignCombination(coefs,line)
                rows += tmp
        rows.append([1]*conju)
        rows.sort()
        for i,line in enumerate(rows):
            print("rows",i,line)

        res,values = self.getOrthogonalChi(coefs, rows,conju)
        res,values = self.getOrthogonalChi(coefs, res,conju)
        res,values = self.getOrthogonalChi(coefs, res,conju)
        res.sort()
        for i,line in enumerate(res):
            print(i,line)

        combi1 = self.combination2List(res[3:9],2)
        combi2 = self.combination2List(res[9:15],2)

        count = 0 
        for line1 in combi1:
            for line2 in combi2:
                table = res[1:3] + list(line1)
                table += list(line2) + [res[-1]]
                judge = self.checkCharacterTable(table,coefs,N)
                count += 1 
                # print(count)
                if judge:
                    print("characters: ",np.array(table))
                # break
        
        return

    def quintic2Sextic(self,p,q):
        """
        docstring for quintic2Sextic
        p,q:
            rational
        """
        x = Symbol("x")
        arr = [1,8*p,40*p**2,160*p**3]
        arr.append(400*p**4)
        arr.append(512*p**5-3125*q**4)
        arr.append(-9375*p*q**4+256*p**6)
        y = self.getPolynomialValues(arr,x)
        y = y.expand().factor()
        # print(y.factor())
        # y = Poly(y).as_dict()
        return y
    def solvableQuintic(self):
        """
        docstring for solvableQuintic
        """
        p,q = -5,4 
        f = lambda p,q:256*p**5 + 3125*q**4 
        x = Symbol("x")
        arr = [1,0,0,0,p,q]
        y = self.getPolynomialValues(arr,x)
        y = y.expand()
        print(y)
        print(f(p,q))

        p,q = 11,44
        p,q = 20,16

        n = 10
        for p in tqdm(range(-n,n)):
            for q in range(1,n):
                # y = self.quintic2Sextic(p,q)
                res = f(p,q)
                judge,m = self.isSquare(abs(res))
                if judge:
                    
                    y = self.quintic2Sextic(p,q)
                    print(p,q,res,m,y)
        print(factorint(f(95,76)))

        p,q = 11,44
        p,q = 124,496
        res = factorint(f(p,q))
        print(res)
        y = self.quintic2Sextic(p,q)
        print(y)

        res = []
        res.append((x - 3240)*(x**5 + 6561000*x**3 + 10628820000*x**2 + 45199057050000*x - 85077539384400000))
        res.append((x**2 - 1136*x + 323424)*(x**4 - 256*x**3 + 596800*x**2 - 82122496*x + 80343871744))
        res.append((x - 640)*(x**5 + 256000*x**3 + 81920000*x**2 + 68812800000*x - 25585254400000))
        res.append((x - 40)*(x**5 + 1000*x**3 + 20000*x**2 + 1050000*x - 24400000))
        res.append((x - 88)*(x**5 + 176*x**4 + 20328*x**3 + 2001824*x**2 + 182016912*x + 4387146368))
        res.append((x - 40)*(x**5 + 200*x**4 + 24000*x**3 + 2240000*x**2 + 153600000*x + 4505600000))
        res.append((x - 1408)*(x**5 + 2816*x**4 + 5203968*x**3 + 8199471104*x**2 + 11928660344832*x + 4600256389971968))
        res.append((x - 640)*(x**5 + 3200*x**4 + 6144000*x**3 + 9175040000*x**2 + 10066329600000*x + 4724464025600000))
        for line in res:
            print(latex(line))


        return

    def getCharacterRows(self,coefs):
        """
        docstring for getCharacterRows
        """
        n = sum(coefs)
        res = self.getSquareEqn(coefs,n)
        # print(res)
        rows = []
        for line in res:
            if line[0] > 0:
                tmp = self.getSignCombination(coefs,line)
                rows += tmp
        return rows 

    def getCharacterUnits(self,coefs):
        """
        docstring for getCharacterUnits
        """
        n = sum(coefs)
        coefs = [1]*len(coefs)
        res = self.getSquareEqn(coefs,n)
        # print(res)
        arr = []
        for line in res:
            line.sort()
            if line not in arr:
                arr.append(line)

        return arr

    def getGroupDegrees(self,N,n_lim):
        """
        docstring for getGroupDegrees
        """
        n = int((N+0.01)**0.5)
        self.n_lim = n_lim
        factors = self.getAllFactors(N)
        res = []
        for i in factors:
            if i > n:
                break 
            res.append(i)

        res = self.getSquareEqnByFactor(res,N)

        output = []
        for line in res:
            if sum(line) == self.n_lim:
                # print(line,self.getNewForm(line))
                output.append(line)

        return output
    def testCharacterExercise(self):
        """
        docstring for testCharacter
        """
        coefs = [1,2,1,2,2]
        coefs = [1,4,5,5,5]
        coefs = [1,3,3,7,7]
        coefs = [1,2,2,5]
        coefs = [1,3,6,6,8]
        coefs = [1,15,20,12,12]
        rows = self.getCharacterRows(coefs)
        print(rows)

        arr = self.getCharacterUnits(coefs)

        print(arr)

        dims = [1,3,3,4,5]

        x = symbols("x0:16")
        eqns = []
        N    = sum(coefs)
        for i in range(4):
            s = dims[i+1]
            for j in range(4):
                s = s + x[i*4+j]*coefs[j+1]
            eqns.append(s)
            s = 1
            for j in range(4):
                s = s + x[j*4+i]*dims[j+1]
            eqns.append(s)
        print(eqns)


        coefs = [1,1,2,2,3,3]
        rows = self.getCharacterRows(coefs)
        print(rows)
        arr = self.getCharacterUnits(coefs)

        print(arr)

        arr1 = [1,2,4]
        arr2 = [3,5,6]
        s = []
        for i in arr1:
            for j in arr2:
                s.append((i+j)%7)
        print(s)
        # a+b = -1, ab = 2 

        coefs = [1,3,3,7,7]
        a = (-1+3**0.5)/2 
        b = -1 - a
        x = (-1+7**0.5)/2 
        y = -1 - x
        res = [[1,1,1,1,1],
               [1,1,1,a,b],
               [1,1,1,b,a],
               [3,x,y,0,0],
               [3,y,x,0,0]]
        res = self.checkCharacterTable(res,coefs,21)
        print(res)

        N = 168
        n_lim = 6
        res = self.getGroupDegrees(N,n_lim)
        # print(res)
        for line in res:
            print(line,self.getNewForm(line))
        # for n_lim in range(2,20):
        #     res = self.getGroupDegrees(N,n_lim)
        #     if len(res) > 0:
        #         print(n_lim,res)
        coefs = [1,21,42,56,24,24]
        rows = self.getCharacterRows(coefs)
        print(rows)

        return

    def plotArray(self,Y):
        """
        docstring for plotArray
        """
        N = len(Y[0])
        x = np.arange(N)*0.001
        labels = ["P","Q","R"]
        plt.xlabel("time")
        for i in range(3):
            plt.plot(x,Y[i,:],lw=2,label=labels[i])
        plt.legend(loc="upper left")
        filename = "aerodynamics.png" 
        print("write to ",filename)
        
        plt.savefig(filename,dpi=300)

        plt.show()

        return
    def testAerodynamics(self):
        """
        docstring for testAerodynamics
        dP/dt = a1QR + b1
        dQ/dt = a2PR + b2
        dR/dt = a3QP + b3
        a1 + a2 + a3 = 0
        """
        deltaT = 0.001
        N = 10000
        X = np.zeros((3,N))
        X[:,0] = [1.0,1.0,1.0]
        A = np.array([[-1.0,1.0],
                      [-1.0,2.0],
                      [2,3.0]])

        indeces = [[1,2],
                   [0,2],
                   [0,1]]
        for i in range(1,N):
            for j in range(3):
                ii,jj = indeces[j]
                res = A[j,0]*X[ii,i-1]*X[jj,i-1] + A[j,1]
                X[j,i] = X[j,i-1] + deltaT*res 

        # print(X[:,:1000])
        self.plotArray(X)
        
        return

    def testXX(self):
        """
        docstring for testXX
        S&=&\int_{0}^{\infty}x^{-x}dx
        """
        x = np.linspace(0.01,10,1000)
        y = x**(-x)
        y = x*np.log(x)
        # plt.plot(x,y)
        # plt.show()

        f = lambda x: x**(-x)
        # f = lambda x: x*np.log(x)
        s,err = inte.quad(f,0,1)
        # s =  1.9954559575000368 error is  3.971688089521308e-09
        print("s = ",s,"error is ",err)

        x = np.arange(1,15)
        print(sum(1/(x**x)))

        x = Symbol("x")
        s = diff(x**(-x),x)
        value = s.subs(x,1)
        res = [1,value]
        for i in range(5):
            s = diff(s,x)
            value = s.subs(x,1)
            print(value,",")

        return

    def testABElliptic(self):
        """
        docstring for testABElliptic
        """
        p   = 2 

        f = lambda x: (1+x**4)**(-1/2)
        s1,err = inte.quad(f,0,1)
        print("s1 = ",s1)

        f = lambda x: (1 + (p-1)*math.cos(x)**2)**(-1/2)
        s2,err = inte.quad(f,0,np.pi/2)
        print("s2 = ",s2,s1*2**0.5)

        a,b = 1,p**0.5 
        # a,b = 1,1.01 
        for i in range(30):
            A = (a+b)/2 
            B = (a*b)**0.5 
            a,b = A,B 
        print("a,b = ",a,b)

        f = lambda x: (1+(p-1)*math.cos(x)**2)**(1/2)
        s3,err = inte.quad(f,0,np.pi/2)
        print(s3,s3*2/np.pi)

        f = lambda x: (1-x**4)**(-1/2)
        s4,err = inte.quad(f,0,1)
        print(s4)

        N = np.arange(1,14)
        print(sum((N+1)/(N**N)))

        
        return

    def getPermArray(self,n,m):
        """
        docstring for getPermArray
        n,m:
            integers with n >= m 
        return:
            1d array
            [1,n,n-1,...]
        """

        res = [1]
        for i in range(m-1):
            num = res[-1]*(n-i)
            res.append(num)
        
        return res
    def getXXTaylor(self):
        """
        docstring for getXXTaylor
        taylor expansion of (x+1)^{-x-1} at x = 0 
        1,-(x+1)x,(x+1)(x+2)x^2

        [1,2],[2,4],[3,6],[4,8],[5,10],[6,12]
        T(0) = [1,1]
        T(1) = [2,3,1]
        sigma(n) = 1 for even, -1 for odd
        2n: K(2n,n)*sigma(n)*T(n-1,n)+K(2n,n+1)*sigma(n+1)*T(n,n-1)
        + K(2n,2n)*sigma(2n)*T(2n-1,0)
        2n+1: sigma(n)*T(n+1,n-1)+sigma(n+1)*T(n,-2)
        + sigma(2n)*T(2n,0)

        """
        N = 200
        res = [1,1]
        print(res)
        total = []
        total.append(self.inverseArray(res))
        for i in range(N):
            res = self.polynomialMulti(res,[1,i+2])
            total.append(self.inverseArray(res))
            # print(i+2,total[-1])

        results = [1]
        
        for n in range(1,N):
            num = 0
            sigma = (-1)**n 
            begin = (n+1)//2 - 1 
            perm  = self.getPermArray(n,n-begin)
            for i in range(begin,n):
                num  += perm[-1+begin-i]*sigma*total[i][n-1-i]
                sigma = - sigma
            results.append(num)
            print(n,num)

        return
    def testPseudoprimes(self,a=1):
        """
        docstring for testPseudoprimes
        L_p = x1^n+x2^n
        x^2 = x + 1 
        x^2 = ax + 1 
        """
        L = [2,1]
        L = [2,a]
        for i in range(1000):
            # num = L[-1] + L[-2]
            num = a*L[-1] + L[-2]
            L.append(num)
            p2 = (not isprime(i+2))
            if L[-1]%(i+2) == 1 and p2:
                # print("a = ",a,i+2)
                break
        
        return i+2

    def testPseudo(self):
        """
        docstring for testPseudo
        """
        # when res = 26
        dicts = {}
        num = 130*2
        for a in tqdm(range(1,num)):
            res = self.testPseudoprimes(a=a)
            # print(a,res)
            if res == 26:
                # print("%5d%5d"%(a,a%130))
                key = str(a // 130)
                if key in dicts:
                    dicts[key].append(a%130)
                else:
                    dicts[key] = [a%130]

        arr = [5,21,31,99,109,125]
        for key in dicts:
            line = arr.copy()
            for i in dicts[key]:
                line.remove(i)
            print(key,line)


        return

    def testLucasSequence(self):
        """
        docstring for testLucasSequence
        x^2 = ax + 1 
        a_{n+1} = a*a_n + a_{n-1}
        a0 = 2 
        a1 = a 
        """

        res = [[2],[1]]
        for i in range(26):
            line = [1]
            for j in range(1,len(res[-1])):
                num = res[-1][j]+res[-2][j-1]
                line.append(num)
            if len(res[-1]) == len(res[-2]):
                line.append(res[-2][-1])

            k = 3 
            if len(line) > k:
                print(i+2,Integer(line[k-1])/line[k],line)
            res.append(line)

        n = 26
        arr = np.array(res[n])
        print(arr%n)    
        for i in [5,21,31,99,109,125]:
            print(i**8%n)    
        return

    def testCoinFountain(self):
        """
        docstring for testCoinFountain
        https://mathpages.com/home/kmath052.htm
        three coins in the fountain
        head for 1, tail for 0
        7 coins 
        0 1 2 
        3 4 5
          6
        circle 1: 0,1,3,4
        circle 2: 1,2,4,5
        circle 3: 3,4,5,6

        """
        circles = [[0,1,3,4],
                   [1,2,4,5],
                   [3,4,5,6]]

        line = [1]*7 
        status = [line]
        rands = np.random.randint(0,3,10000)
        for i in rands:
            line = line.copy()
            for j in circles[i]:
                line[j] = 1 - line[j]
            if line not in status:
                status.append(line)
        # print(len(status),status)
        string = "%d %d %d\n%d %d %d\n  %d  "
        for i,line in enumerate(status):
            print("------%d------"%(i))
            print(string%(tuple(line)))

        
        return

    def testFiboFrac(self):
        """
        docstring for testFiboFrac
        """
        res = [0,1,1]
        para = [1,1,1]
        a = 0 
        k = 20
        frac = 1/k 
        for i in range(1000):
            num = 0
            for j in range(3):
                num += para[j]*res[-j-1]
            res.append(num)
        # print(res[-1])
        for i in res:
            frac = frac/k
            a = a + i*frac
        num = k*(k*(k-para[0])-para[1])-para[2]
        print(a,a*num) 

            
        return

    def testTaylorSeries(self):
        """
        docstring for testTaylorSeries
        """
         
        arr = [[3,3],
              [5,30],
              [6,90],
              [7,630],
              [9,22680],
              [10,113400],
              [11,1247400],
              [13,97297200],
              [14,681080400],
              [15,10216206000]]

        for i,j in arr:
            print(i,j,factorial(i)/j)

        arr = [[4,3,3],
               [89,5,120],
               [83,6,144],
               [593,7,1260],
               [287,8,720],
               [41891,9,120960],
               [158869,10,518400],
               [152447,11,554400],
               [678871,12,2721600],
               [473847547,13,2075673600],
               [1411410779,14,6706022400],
               [63830876233,15,326918592000],
               [21647448733,16,118879488000],
               [2427359621209,17,14227497123840],
               [4029579070333,18,25107347865600],
               [209443167797947,19,1382330686464000]]
        for k,i,j in arr:
            print(i,j,k*factorial(i)/j)
            # break 

        # log(1+x)*exp(-x)
        print("----------------------------")
        print("----------------------------")
        n = 20
        logx = [0,1]
        sigma = -1
        for i in range(n):
            logx.append(sigma*Integer(1)/(i+2))
            sigma = -sigma
        print(logx)

        expx = [Integer(1)]

        for i in range(n+1):
            expx.append(-expx[-1]/(i+1))

        print(expx)

        res = [logx[0]*expx[0]]
        for i in range(1,n+1):
            num = 0 
            for j in range(i+1):
                num += logx[j]*expx[i-j]
            num = factorial(i)*num
            res.append(num)
            print(i,num)

        return  


    def testLucasSum(self):
        """
        docstring for testLucasSum
        S&=&\sum_{k=0}^{n}\frac{L_{2k+1}}{\left(2k+1\right)^{2}C_{2k}^{k}}
        """
        lucas = [Decimal(-1),Decimal(2)]
        for i in range(100):
            num = Decimal(sum(lucas[-2:]))
            lucas.append(num)
        # print(lucas)
        # from decimal import *
        getcontext().prec = 100
        res = Decimal(0)
        c2nn = Decimal(1)
        for i in range(20):
            n = Decimal(2*i + 1)
            res += lucas[2*i+1]/(n*n)/c2nn 
            # print(i,c2nn)
            c2nn = c2nn*(n+1)*n/((i+1)**2)

        print(res)
        return  

    def testIntegral(self):
        """
        docstring for testIntegral
        """
        f = lambda x: np.exp(np.sin(x))*np.sin(x)
        s4,err = inte.quad(f,0,2*np.pi)
        print(s4)

        res = 0 
        n_factorial = 1
        for i in range(50):
            n_factorial = n_factorial/((i+1)**2)
            res += n_factorial*(2*i+2)/(2**(2*i+1))
        print(res,res*np.pi)

        f = lambda x: (np.sin(x)*np.cos(x))**2/(1-x**2)**0.5
        s4,err = inte.quad(f,-0.5,0.5)
        print(s4)

        # series expansion
        res = 0
        return

    def getTaylor(self,x,f,n = 10,step=1):
        """
        docstring for getTaylor
        f:
            a function
        """
        print("--------",f)
        print(0,f.subs(x,0))
        count = 0 
        for i in range(n):
            f = diff(f,x,step)
            count += step
            print(count,f.subs(x,0))
        return
    def testExpCos(self):
        """
        docstring for testExpCos
        """
        x = Symbol("x")
        f = exp(exp(x)-1)
        self.getTaylor(x,f)
        self.getTaylor(x,exp(sin(x)),n=2)
        self.getTaylor(x,exp(1-cos(x)),n=10)
        f = atan(sqrt(1+x*x)-1)
        self.getTaylor(x,f,n=3)

        a = Integer(33399969978375)/factorial(18)
        a = Integer(4368604540935009375)/factorial(22)
        print(a)

        arr = [[2 ,1],
              [4 ,-3],
              [6 ,15],
              [8 ,-315],
              [10, 36855],
              [12, -4833675],
              [14, 711485775],
              [16, -133449190875],
              [18, 33399969978375],
              [20, -10845524928112875],
              [22, 4368604540935009375],
              [24, -2121018409773134746875],
              [26, 1222083076784378918484375],
              [28, -826013017674132244878796875],
              [30, 647724113841936142199672859375],
              [32, -583169643524919352829283528046875],
              [34, 597359177463144491308077497692734375],
              [36, -690705634748275077389006998731704296875],
              [38, 895308770935086347664096961848512087109375],
              [40, -1293014756515190918978281765759779961360546875],
              [42, 2069090654296251423055420621941125742237708984375],
              [44, -3650287422951500638665261346616153988543455841796875],
              [46, 7067644991767194875269793917621503636445835249443359375]]
        for i,j in arr:
            a = Integer(abs(j))/factorial(i)
            nom,denom = fraction(a)
            # print(i,denom)
            # print(i,denom,factorint(denom/i))
            print(i,denom,factorint(denom))
            break

        f = (1+x)**(1+x)
        self.getTaylor(x,f,n=2)

        arr = [[0, 1],
              [1, 1],
              [2, 2],
              [3, 3],
              [4, 8],
              [5, 10],
              [6, 54],
              [7, -42],
              [8, 944],
              [9, -5112],
              [10, 47160],
              [11, -419760],
              [12, 4297512],
              [13, -47607144],
              [14, 575023344],
              [15, -7500202920],
              [16, 105180931200],
              [17, -1578296510400],
              [18, 25238664189504],
              [19, -428528786243904],
              [20, 7700297625889920],
              [21, -146004847062359040],
              [22, 2913398154375730560],
              [23, -61031188196889482880],
              [24, 1339252684282847781504],
              [25, -30722220593745761750400],
              [26, 735384500710300207353600],
              [27, -18335978741646751132022400],
              [28, 475481863771471289192140800],
              [29, -12804628568422088117232979200],
              [30, 357611376476800486783526273280]]

        for i,j in arr:
            a = Integer(abs(j))/factorial(i)
            nom,denom = fraction(a)
            # print(i,denom)
            # print(i,denom,factorint(denom/i))
            print(i,nom,denom,factorint(denom))


        f = asin(1-sqrt(1-x*x))
        
        # f = asin(sqrt(1+x*x)-1)
        self.getTaylor(x,f,n=2)

        arr = [[2,1],
               [4,3],
               [6,60],
               [8,2205],
               [10,150255],
               [12,15592500],
               [14,2329051725],
               [16,470583988875],
               [18,123830632926000],
               [20,41119444417676625]]
        for i,j in arr:
            print(i,factorint(j))

        f = exp(atan(x))
        self.getTaylor(x,f,n=5)

        f = atan(1-sqrt(1-x*x))
        self.getTaylor(x,f,n=3)

        return

    def judgeRamaMagic(self,magic_square):
        """
        magic_square:
            2d array of n*n
        return:
            yes or no
        """

        n = len(magic_square)
        magic_square = np.array(magic_square)
        total = sum(magic_square[0])

        # n rows, n columns, tow diagonals
        arr_sum = np.zeros(2*n+2,int)
        arr_sum[:n] = np.sum(magic_square,axis=0)
        arr_sum[n:2*n] = np.sum(magic_square,axis=1)
        for i in range(n):
            arr_sum[2*n] += magic_square[i,i]
            arr_sum[2*n+1] += magic_square[i,n - 1 - i]

        # print(arr_sum)

        combinations = itertools.combinations(np.arange(n),2)
        arr = []
        for line in combinations:
            arr.append(list(line))

        res = []
        count = 0 
        for row in arr:
            for col in arr:
                matrix = np.zeros((n,n),int)
                num = 0 
                for i in row:
                    for j in col:
                        tmp = magic_square[i,j] 
                        matrix[i,j] = tmp 
                        num += tmp
                if num == total:
                    count += 1
                    res.append([row,col])
                    # print(count,"-------",row,col)
                    # print(matrix)
        # print(len(res),res)
        print("equations number",len(res))

        line = [[0,2],[1,3]]
        res = [0,0]
        for i,j in line:
            res[0] += magic_square[i,j]
            res[0] += magic_square[j,i]
            res[1] += magic_square[i,n-1-j]
            res[1] += magic_square[j,n-1-i]
        # print(res)




        return
        
    def testRamaMagicSquare(self):
        """
        docstring for testRamaMagicSquare
        """
        res = [[23, 11, 19, 90],
               [91, 18, 8, 26],
               [9, 25, 92, 17],
               [20, 89, 24, 10]]


        # self.judgeRamaMagic(res)

        # Ramanujan's birthday: 1887-12-22
        # 86 87 88 89
        # 9  10 11 12
        # 16 17 18 19
        # 22 23 24 25
        res = [[22, 12, 18, 87],
               [88, 17, 9, 25],
               [10, 24, 89, 16],
               [19, 86, 23, 11]]


        # self.judgeRamaMagic(res)
        for i in range(100,101):
            line = [50,2,3,i]
            num = self.checkRamaMagic(line)
            print(line,"number = ",num)
            break
        num = self.checkRamaMagic([1,2,16,15])


        return  
    def checkSuit(self,checked, total):
        return (checked in total) or (checked <= 0) or (checked >= 18)

    def checkRamaMagic(self,arr):
        """
        check Ramanujan matrix 
        code from the internet
        https://cloud.tencent.com/developer/article/1503127
        (a, b, c, d) = (22, 12, 18, 87)
        (e, f, g, h) = (0, 0, 0, 0)
        (i, j, k, l) = (0, 0, 0, 0)
        (m, n, o, p) = (0, 0, 0, 0)

        """
        count = 0
        a, b, c, d = arr
        x = 4*max(arr)
        y = sum(arr)

        for f in range(1, x):
            for k in range(1, x):
                line = [a, b, c, d, f]
                if self.checkSuit(k, line):continue
                line.append(k)
                p = y - a - f - k
                if self.checkSuit(p, line):continue
                line.append(p)
                e = y - a - b - f
                if self.checkSuit(e, line):continue
                line.append(e)
                m = y - a - d - p
                if self.checkSuit(m, line):continue
                line.append(m)
                i = y - a - e - m
                if self.checkSuit(i, line):continue
                line.append(i)
                j = y - e - f - i
                if self.checkSuit(j, line):continue
                line.append(j)
                n = y - b - f - j
                if self.checkSuit(n, line):continue
                line.append(n)
                o = y - m - n - p
                if self.checkSuit(o, line):continue
                line.append(o)
                g = y - c - k - o
                if self.checkSuit(g, line):continue
                line.append(g)
                h = y - e - f - g
                if self.checkSuit(h, line):continue
                line.append(h)
                l = y - d - h - p
                if self.checkSuit(l, line):continue
                line.append(l)
                if ((y == c + d + g + h) and (y == i + j + m + n) and \
                    (y == k + l + o + p) and (y == i + j + k + l) and \
                    (y == b + c + n + o) and (y == e + i + h + l) and \
                    (y == d + g + j + m) and (y == f + g + j + k) and \
                    (y == b + e + l + o) and (y == c + h + i + n) and \
                    (y == g + h + k + l)):
                    count += 1
                    line = np.array([ a, b, c, d,
                                      e, f, g, h,
                                      i, j, k, l,
                                      m, n, o, p])
                    # line = line.reshape((4,4))
                    line.sort()
                    # print(count,line)
                    # print("count:",count,line)
                    # self.judgeRamaMagic(line)
        # print("numbers",count)
        return count

    def testArctanSqrt(self):
        """
        docstring for testArctanSqrt

        f\left(x\right)&=&\arctan\left(1-\sqrt{1-x^{2}}\right)

        """

        paras = [Integer(1),Integer(1)/2]
        n = 3000
        for i in range(n):
            num = paras[-1]*(2*i+2)*(2*i+1)
            num = num /(4*(i+2)*(i+1))
            paras.append(num)
        # print(paras)

        b = []
        b.append(paras[0])
        b.append(-paras[1])
        for i in range(2,n+2):
            num = paras[i-1] - 3*paras[i]
            b.append(num)


        # print(b)

        factorials = self.getFactorials(2*n+1)
        factorials = [1] + factorials

        res = [1]
        factors = []
        # for i in tqdm(range(1,n)):
        for i in range(1,n):
            num = 0
            for k in range(i):
                # print("k = ",k,res[k],b[i-k])
                num += res[k]*b[i-k]/factorials[2*k+1]

            p,q = fraction(-num/(2*i+2))
            num = -num*factorials[2*i+1]
            a = factorint(q)[2]
            k = 2*(i+1)-a
            # print(i+1,a,k,factorint(2*(i+1)),factorint(q))
            p,q = fraction(q/(2*i+2))
            # print(i+1,q)
            if q != 1:
                print(i+1,q)
            # print(i+1,isprime(p))
            # if isprime(p):
            #     print(i+1,p)
            factors.append(k)
            # print(i+1,q)
            # print(i,num)
            res.append(num)

        # print(factors)

        print("get factors")
        res = [1]
        for i in range(2):
            line = []
            for j in res:
                line.append(j+1)
            res = res + line 
        print(res)



        
        return

    def checkFermatPrime(self,n,m,num=6):
        """
        docstring for checkFermatPrime
        n,m:
            integers
        return:
            [2,1] ==> [3,5,17,257,65537,...]
        """
        results = []
        for i in range(num):
            res = n**(2**i)+m
            res = isprime(res)
            results.append(res)
        return results 
    def testFermatNum(self):
        """
        docstring for testFermatNum
        May I conjecture that there are 
        no more than 5 k-s so that p(n) 
        are primes for any pair of (n,m)?
        """
        n = 500
        p = []
        for i in range(2,n+2):
            p.append(prime(i))
        # print(p)

        dicts = {}
        overFour = {}
        length = 7 
        for i in range(4,length+1):
            overFour[i] = []

        for q in tqdm(p):
            for i in range(2,q,2):
                res = self.checkFermatPrime(q,i,num=length)
                k = sum(res)
                if k in dicts:
                    dicts[k] += 1 
                else:
                    dicts[k]  = 1
                # print(q,i,k,res)
                if k > 3:
                    overFour[k].append([q,i])
                    print(q,i,k)

        print(dicts)
        print(overFour)
        return

    def getExponentString(self,factors,exponents):
        """
        docstring for getExponentString
        """
        res = ""
        for i,j in zip(factors,exponents):
            if j == 1:
                res = "%s%d*"%(res,i)
            elif j > 1:
                res = "%s%d^{%d}*"%(res,i,j)
        res = res[:-1]
        return res

    def getFactorString(self,factors,exponents,splitNum):
        """
        docstring for getFactorString
        factors:
            1d array
        exponents:
            1d array 
        [2,3,5,7],[3,5,2,4],2 
          => 2^{3}*3^{5} + 5^{3}*3^{5}
        """
        res1 = self.getExponentString(factors[:splitNum],
                                     exponents[:splitNum])
       
        res2 = self.getExponentString(factors[splitNum:],
                                     exponents[splitNum:])

        res = "%s + %s"%(res1,res2)
        
        return res

    def ABCTest(self,line1,line2,n = 10):
        """
        docstring for ABCTest
        """
        res = np.arange(n)
        
        len1  = len(line1)
        len2  = len(line2)
        length = len1 + len2
        combinations = itertools.product(res,repeat=length)

        for line in combinations:
            if sum(line[:len1]) == 0:
                continue
            if sum(line[len1:]) == 0:
                continue
            radical = 1 
            for i,j in enumerate(line1):
                if j > 0:
                    radical *= line1[i]
            for i,j in enumerate(line2):
                if j > 0:
                    radical *= line2[i]

            a = 1 
            b = 1
            for i,j in zip(line1,line[:len1]):
                a *= i**j
            for i,j in zip(line2,line[len1:]):
                b *= i**j  

            c = a + b

            dicts = factorint(Integer(c))
            count = 0 
            for factor in dicts:
                count += 1 
                radical = radical*factor 
            if radical < c and count == 1:
                # print(line,a,b,c,radical)
                string = self.getFactorString(line1+line2,line,len1)
                i = factor 
                j = dicts[i]
                print("%s &=& %d^{%d} \\\\"%(string,i,j))
        return
    def testABCConjecture(self):
        """
        docstring for testABCConjecture
        """
        primes = []
        for i in range(1,200):
            primes.append(prime(i))

        # self.ABCTest(line1,line2)
        combinations = itertools.combinations(primes,2)
        combi = [[0,1,2,3],
                 [0,2,1,3],
                 [0,3,1,2]]

        for line in tqdm(combinations):
            self.ABCTest([line[0]],[line[1]],n=6)
            continue
            for indeces in combi:
                line1 = []
                for i in indeces:
                    line1.append(line[i])
                self.ABCTest(line1[:2],line1[2:])


        return

    def getCollatzNum(self,num):
        """
        docstring for getCollatzNum
        """
        res = 0 
        line = [num]
        while num > 1:
            if num%2 == 0:
                num = num//2 
            else:
                num = 3*num + 1 
            res += 1 
            line.append(num)
        print(line)

        return res
    def testCollatz(self):
        """
        docstring for testCollatz
        """
        res = 1
        y = []
        x = []
        for i in range(1):
            res = 9*res + 1 
            y.append(self.getCollatzNum(res))
            x.append(i+1)
            print(x[-1],y[-1])
        plt.plot(x,y)
        # plt.savefig("collatz.png",dpi=300)
        # plt.show()
        n = 7
        print(self.getCollatzNum(2**n-1))

        res = [1,2]
        for i in range(10):
            num = res[-1]*res[-2]-1
            print(num**0.5)
            num = 7*res[-1] - res[-2]
            res.append(num)

        res = [Integer(2),Integer(2)]
        for i in range(10):
            a,b = res[-2:]
            num = 2*b - 3*a*b + 17*a - 16
            num /= (3*b - 4*a*b + 18*a - 17)
            res.append(num)
            print(i+3,sqrt(1/(num-1)))


        return

    def tangentPower(self):
        """
        docstring for tangentPower
        """
        
        for n in range(1,10):
            num  = bernoulli(2*n+2)
            a    = 2**(2*n+1)
            num *= a*(2*a-1)*(2*n+1)*(2*n)/factorial(2*n+2)
            num  = abs(num)
            k    = 2*n
            num2 = 2**k*(2**k-1)*bernoulli(k)/factorial(k)
            print(2*n-1,num,num-abs(num2))
        return

    def getSeriesT(self,k,n=1000,zeta = False):
        """
        docstring for getSeriesT
        Tk = 1-1/3^k+1/5^k+...
        or 
        zeta(k)
        """
        res = 0 
        if zeta is False:
            for i in range(1,n):
                res += (-1)**(i-1)/(2*i-1)**k
        else:
            for i in range(1,n):
                res += 1/i**k

        return res

    def arctanIntegral(self):
        """
        docstring for arctanIntegral
        \int_0^1 (arctan x)^n dx
        """
        print(self.getSeriesT(2,n=100))

        f = lambda x: (np.arctan(x))**3 
        res = inte.quad(f,0,1)[0]
        print("area",res)

        T2 = self.getSeriesT(2)
        T3 = self.getSeriesT(3)
        Z3 = self.getSeriesT(3,zeta=True)
        print(T3,Z3)
        print(np.pi**3/T3)
        print(np.pi**5/self.getSeriesT(5))

        res = np.pi**3/64+3*np.pi**2*np.log(2)/32 
        res += (-3*np.pi*T2/4 + 63*Z3/64)
        print("area",res)

        f = lambda x: np.log(1+x**2)*(np.arctan(x))/(1+x**2)
        res = inte.quad(f,0,1)[0]
        print("integral",res)
        return

    def getEulerNumbers(self):
        """
        docstring for getEulerNumbers  
        S(s)&=&S\left(s,\frac{1}{2}\right)
        =\sum_{k=0}^{\infty}\frac{\left(-1\right)^{k}}{\left(2k+1\right)^{s}}
        """
        res = [Integer(1)/4]
        for n in range(1,100):
            num = 1/factorial(2*n)/2**(2*n+1)
            for k in range(n):
                num -= (-1)**k*res[k]/factorial(2*n-2*k)
            num = num*(-1)**n/2 
            res.append(num)
            a,b = fraction(num)
            # print(n,num,factorint(a))
            if isprime(a):
                print(n,a)

        return

    def getClosestPoints(self,points):
        """
        docstring for getClosestPoints
        """
        output = [points[0]]
        num = len(points)
        for i in range(1,num):
            p2 = points[i]
            judge = 1 
            # compare every point in the output
            for p1 in output:
                res = 0 
                for i,j in zip(p1,p2):
                    res += (i - j)**2 
                if res != 1:
                    judge = 0 
                    break
            if judge == 1:
                output.append(p2)

        return output

    def sphericalPacking(self):
        """
        docstring for sphericalPacking
        """
        points = []
        res = [-1,1]
        combi = itertools.product(res,repeat=1)
        for line in combi:
            p = [0,0,line[0]*sqrt(6)/4,sqrt(10)/4]
            points.append(p)
            p = [0,sqrt(3)/3,line[0]*sqrt(6)/12,sqrt(10)/4]
            points.append(p)
            p = [Integer(1)/2,-sqrt(3)/6,line[0]*sqrt(6)/12,sqrt(10)/4]
            points.append(p)
            p = p.copy()
            p[0] = -p[0]
            points.append(p)

        points = self.getClosestPoints(points)
        # print(points,len(points))
        for i,p in enumerate(points):
            print(i,p)

        
        return

    def thetaSeries(self):
        """
        docstring for thetaSeries
        """
        q = Symbol("q")
        th2 = q**2*(1+2*q**2+2*q**6+2*q**12)**8
        th3 = (1+2*q+2*q**4+2*q**9)**8
        th4 = (1-2*q+2*q**4-2*q**9)**8
        s = th2 + th3 + th4 
        s = s.expand()

        # Leech lattice
        tau = [0,1,-24, 252, -1472, 4830, -6048, 
               -16744, 84480, -113643, -115920, 534612, 
               -370944, -577738, 401856]
        sigma11 = self.getSigmaN(11,n = 10)
        for i in range(2,10):
            num = sigma11[i] - tau[i]
            num = num*65520//691
            print(i,num)


        return

    def getVolume3(self,A):
        """
        docstring for getVolume3
        A:
            list of length 3, [1,2,3]
        """
        types = "\\theta_{%d%d}"
        angles = [Symbol(types%(A[0],A[1])),
                  Symbol(types%(A[1],A[2])),
                  Symbol(types%(A[0],A[2]))]

        mat = Matrix.ones(3)
        line = [[0,1],[0,2],[1,2]]
        for i,(k,j) in enumerate(line):
            mat[k,j] = cos(angles[i])
            mat[j,k] = cos(angles[i])
        
        return mat.det()

    def testPolytope(self):
        """
        docstring for testPolytype
        """
        n = 4
        mat = Matrix.ones(n)
        texts = []
        for i in range(n):
            texts.append(str(i+1))
        index = 0
        for i in range(n):
            for j in range(i+1,n):
                angle = "\\theta_{%s%s}"%(texts[i],texts[j])
                angle = Symbol(angle)
                mat[i,j] = cos(angle)
                mat[j,i] = mat[i,j]
                index += 1
        # print(latex(mat))
        v4 = mat.det()
        v1 = self.getVolume3([1,2,3])
        v2 = self.getVolume3([1,2,4])
        s = v1*v2 -v4 
        s = s.expand().trigsimp()

        s = latex(v4)
        s = s.replace("{\\left(","")
        s = s.replace("\\right)}","")
        # print(s)

        a = sqrt(5)
        s = (105+47*a)/(30+14*a)
        s = 3*(1+a)**2+(7+3*a)**2
        # print(s.simplify()/8)

        alpha,beta,gamma = symbols("alpha, beta, gamma")
        angles = [alpha,beta,gamma]
        angles = [alpha,beta]*3 
        angles = [alpha,alpha,alpha,beta,beta,beta]
        angles = [alpha,alpha,alpha,alpha,beta,alpha]
        line = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
        for i,(k,j) in enumerate(line):
            mat[k,j] = cos(angles[i])
            mat[j,k] = cos(angles[i])
        s = latex(mat.det())
        s = s.replace("{\\left(","")
        s = s.replace("\\right)}","")
        print(s)

        return

    def getPhiCode(self,A,x):
        """
        docstring for getPhiCode
        phi = ax^2+bx+c
        """
        a,b,c = A 
        res1 = self.mulTable[a][x]
        res1 = self.mulTable[res1][x]
        res2 = self.mulTable[b][x]

        res = self.addTable[res1][res2]
        res = self.addTable[res][c]
        
        return res 

    def getHexacode(self,A,number=False):
        """
        docstring for getHexacode
        """
        res = A.copy()

        for i in range(1,4):
            res.append(self.getPhiCode(A,i))

        if number:
            res = str(res).replace("[","")
            res = str(res).replace("]","")
            res = str(res).replace(" ","")
            res = str(res).replace(",","")
            return str(res)
        string = ""
        for i in res:
            string += self.codes[i]

        return string

    def getHexaImages(self,a):
        """
        docstring for getHexaImages
        """
        res = np.arange(2)
        combi = itertools.product(res,repeat=3)

        images = []
        for line in combi:
            indeces = []
            for i in line:
                indeces.append(i)
                indeces.append(1-i)
            item = ""
            for i in range(3):
                for j in range(2):
                    index = indeces[2*i+j]
                    item += a[2*i+index]
            # print(indeces,item)
            if item in self.hexa:
                images.append(item)
        print(images)
        return images

    def getSameHexaLen(self,code1,code2):
        """
        docstring for getSameHexaLen
        """
        res = 0 
        for i,j in zip(code1,code2):
            if i == j:
                res += 1 
        return res

    def hexaCode(self):
        """
        docstring for hexaCode
        filed F4 0,1,\omega,\bar{\omega}
        simplified by 0,1,2,3

        """
        self.addTable = [[0,1,2,3],
                         [1,0,3,2],
                         [2,3,0,1],
                         [3,2,1,0]]

        self.mulTable = [[0,0,0,0],
                         [0,1,2,3],
                         [0,2,3,1],
                         [0,3,1,2]]

        self.codes = ["0","1","\\omega","\\bar{\\omega}"]

        res = np.arange(4)
        combi = itertools.product(res,repeat=3)

        dicts = {}

        hexa = []
        for A in combi:
            res = self.getHexacode(list(A),number=True) 
            hexa.append(res)

        # print(hexa)
        self.hexa = hexa

        # classification
        a = "232323"
        # a = "010123"
        self.getHexaImages(a)

        res = np.arange(64)
        combi = itertools.combinations(res,2)
        dicts = {0:0,1:0,2:0}
        for line in combi:
            i,j = line 
            num = self.getSameHexaLen(hexa[i],hexa[j])
            dicts[num] += 1 
        # print(dicts)        
        
        return

    def sphericalCrown(self):
        """
        docstring for sphericalCrown
        """
        R,h,x = symbols("R h x")

        m = 3
        v = integrate((R**2-x**2)**m,[x,R-h,R])
        v = v + (h*(2*R-h))**m*(R-h)/(2*m+1)
        v = v.factor()

        print(latex(v))

        return
    
    def getGCoef(self):
        """
        docstring for getGCoef
        """

        alpha = 7
        getSum = lambda n: sum([2**(alpha*k) for k in range(n+1)])

        coefs = [1]
        for i in range(1,7):
            res = -1 
            for j in range(i):
                res = res - coefs[j]*getSum(i-j)
            coefs.append(res)
            print(i,res)

        res = np.arange(100)*2 + 1 
        # print(res)
        k = 2
        a = sum(res**k/(1+np.exp(res*np.pi)))
        print(1/a)
        return


    def justifyGolay(self,index):
        """
        docstring for justifyGolay
        res:
            2d array with size [4,6]
        """
        hexaList = lambda i:np.array(list(self.hexa[i])).astype(int)
        golay = hexaList(index)
        print(golay)

        count = 0 
        counts = {8:0,12:0,16:0}
        justify = [[1,3],[2,4]]
        for index in range(2):
            for i in range(128):
                order = []
                k = i 
                for j in range(6):
                    order.append(k%2 + 2*index)
                    k = k // 2
                res = []
                # print(i,order)
                for j,k in enumerate(golay):
                    code = self.oddEvenTable[k,order[j]]
                    res.append(code)
                res = np.array(res).transpose()
                if sum(res[0]) in justify[index]:
                    count += 1
                    num = np.sum(res)
                    # print(i,num)
                    counts[num] += 1
        return count,counts
    def golayCode(self):
        """
        docstring for golayCode
        Golay code is based on hexacode
        There are four interpreations for each 
        digit out of [0,1,omega,omega_], two odd 
        ones and two even ones
        """
        oddEvenTable = [[[1,0,0,0],[0,1,1,1],
                         [0,0,0,0],[1,1,1,1]],
                        [[0,1,0,0],[1,0,1,1],
                         [1,1,0,0],[0,0,1,1]],
                        [[0,0,1,0],[1,1,0,1],
                         [0,1,0,1],[1,0,1,0]],
                        [[0,0,0,1],[1,1,1,0],
                         [1,0,0,1],[0,1,1,0]]]

        self.oddEvenTable = np.array(oddEvenTable)

        # print(oddEvenTable[0,2])
        # get hexacode
        self.hexaCode()
        binary  = lambda x:np.array(list(bin(x)[2:])).astype(int)
        print(self.hexa[10:20])


        total = {8:0,12:0,16:0}
        for i in range(64):
            count,counts = self.justifyGolay(i)
            for key in counts:
                total[key] += counts[key]

            print(i,"count",count,counts)
        print(total)
        
        return

    def selectFullSequence(self,fullSeq,num):
        """
        docstring for selectFullSequence
        """
        res = fullSeq[0]
        residue = fullSeq.copy()
        residue.remove(fullSeq[0])

        count = 0 
        for line in fullSeq:
            judge = 1
            for i in line:
                if i in res:
                    judge = 0
                    break 
            if judge:
                res += line 
                residue.remove(line)
            if len(res) == num*3:
                break
            count += 1 
        print("count = ",count)

        return res,residue

    def steinerSystem(self):
        """
        docstring for steinerSystem
        """
        tuples = [[4,5],[5,6],[3,6],
                  [4,7],[5,8]]
        t,k = tuples[0]
        combi = lambda n,m: factorial(n)/(factorial(n-m))/factorial(m)
        kt = combi(k,t)
        for v in range(5,20):
            B = combi(v,t)/kt
            print(v,B)

        # steiner triple systems
        # v = 6*n + 3
        res = []
        n = 2
        num = 2*n+1
        for i in range(num):
            line = []
            for j in range(3):
                line.append(3*i+j+1)
            # print(i,line)
            line.sort()
            res.append(line)

        ij = itertools.combinations(np.arange(num),2)
        # commutative idempotent group
        # 012    021  
        # 120 -> 210  
        # 201    102 

        tran = lambda i,j:(((i+j)%num)*(n+1))%num
        for i,j in ij:
            for k in range(3):
                line = []
                line.append(i*3+k+1)
                line.append(j*3+k+1)
                a = tran(i,j)*3+(k+1)%3+1
                line.append(a)
                line.sort()
                # print(i,j,k,line)
                res.append(line)

        total = []
        for line in res:
            total.append(line[:2])
            total.append(line[1:])
            total.append([line[0],line[2]])

        total.sort()
        # print(total,len(total))

        res.sort()
        # arr1,res = self.selectFullSequence(res,num)

        order = []
        for i in range(num*3):
            order.append(i+1)

        lines = itertools.combinations(np.arange(35),5)
        for line in lines:
            tmp = []
            for i,j in enumerate(line):
                tmp += res[j]
            tmp.sort()
            # print(tmp)
            if tmp == order:
                print(line,tmp)


        # print(combi(24,5)/combi(8,5))

            
        return  

    def steiner45n(self):
        """
        docstring for steiner45n
        """
        p = [3,4,5] 
        P = np.prod(p)
        values = self.getRemainderValues(p)
        values = np.array(values)

        arr = [[0,2],[1,3],[0,1,2,3]]
        lines = self.getAllCombinator(arr)

        res = []
        for line in lines:
            line = np.array(line)
            num = sum(line*values)%(P) 
            if num in res:
                print("repeat",line,num)
                continue
            print(line,num)
            res.append(num)
        res.sort()
        print(res)


        return


    def steiner5824(self):
        """
        docstring for steiner5824
        """
        combi = lambda n,m: factorial(n)/(factorial(n-m))/factorial(m)
        combi2 = lambda t,k,v:combi(v,t)/combi(k,t)

        paras = []
        for k in range(5):
            num = combi(24-k,5-k)/combi(8-k,3)
            print(k,num)
            paras.append(num)
        paras += [1,1,1]
        print(paras)

        res = []
        total = []
        for i in paras:
            line = [i]
            for j in res:
                line.append(j-line[-1])
            res = line 
            print(res)
            total.append(res)

        for res in total:
            res = np.flip(res)
            print(res)

        print(combi(12,5),combi(8,5))
        print(combi2(5,8,24))
        print(combi2(5,6,108))
        
        return


    def origamiCubic(self):
        """
        docstring for origamiCubic
        """
        a,b,x,y = symbols("a,b,x,y")
        s = (1+x) - 9*(1-x)/(2-x)**2 
        s,_ = fraction(s.simplify())
        s = s.expand()

        print(s)
        s = s.subs(x,3/(2-x)-1)
        s,_ = fraction(s.expand().simplify())
        print(s)

        s = (1+x) - (1-x)/(a+b-a*x)**2 
        s,_ = fraction(s.simplify())
        s = s.expand()
        s = s.collect(x)

        # print(latex(s))
        # s = s.subs(x,1/(a+b-a*x)-1)
        # s,_ = fraction(s.expand().simplify())
        # s = s.collect(x)
        # print(latex(s))

        x = self.getCubicSol([1,-3,3,-3])
        # print(latex(x))

        x = self.getCubicSol([b,-1,b+2*a,-1])
        x = self.getCubicSol([b,-1,1/3/b,-1])

        return

    def curvature(self):
        """
        docstring for curvature
        """

        theta,x,y,z = symbols("theta x y z")
        x1 = -x*sin(theta) + y*cos(theta)
        y1 = x*cos(theta) + y*sin(theta)
        x2 = -2*y*sin(theta) + (z-x)*cos(theta)
        y2 = 2*y*cos(theta) + (z-x)*sin(theta)

        s = x1*y2 - x2*y1 
        s = s.expand().trigsimp()
        print(s)

        s = x1**2+y1**2 
        s = s.expand().trigsimp()
        print(s)

        s1 = sqrt(cos(2*theta))
        s2 = diff(s1,theta)
        s3 = diff(s2,theta)

        s = (s1**2 - s1*s3 + 2*s2**2)/sqrt(s1**2+s2**2)**3 
        s = s.expand().trigsimp()
        print(latex(s))
        return

    def diffVector(self,vec,x):
        """
        docstring for diffVector
        """
        res = []

        for s in vec:
            res.append(s.diff(x))

        return res 

    def getSurfaceArea(self,X,u,v):
        """
        docstring for getSurfaceArea
        get the formula of the surface area
        """
        Xu = self.diffVector(X,u)
        Xv = self.diffVector(X,v)

        dotProd = lambda X,Y: sum([x*y for x,y in zip(X,Y)])
        E = dotProd(Xu,Xu)
        F = dotProd(Xu,Xv)
        G = dotProd(Xv,Xv)

        s = E*G - F**2 
        s = s.trigsimp()

        return s
    def surfaceArea(self):
        """
        docstring for surfaceArea
        """
        u,v,r = symbols("u v r")
        X = [r*sin(u)*cos(v),r*sin(u)*sin(v),r*cos(u)]
        s = self.getSurfaceArea(X,u,v)
        print(s)

        X = [cos(u)*cos(v),cos(u)*sin(v),cos(v)**2]
        s = self.getSurfaceArea(X,u,v)
        print(s)

        X = [cos(u)**3*cos(v)**3,cos(u)**3*sin(v)**3,sin(u)**3]
        # s = self.getSurfaceArea(X,u,v)
        # print(latex(s))

        
        
        return

    def spherialTriangle(self):
        """
        docstring for spherialTriangle
        """
        a,b,c = symbols("a b c")

        p = (a+b+c)/2 
        # s1 = sin(p)*sin(p-a)/(sin(c)*sin(b)-(cos(p-a))**2*sin(p-b)*sin(p-c))
        # s2 = sin(p)*sin(p-b)/(sin(c)*sin(a)-(cos(p-b))**2*sin(p-a)*sin(p-c))
        bc = (cos(b-c)-cos(a))/2
        a2 = (1+cos(b+c)*cos(a)+sin(b+c)*sin(a))/2
        s = (sin(c)*sin(b)-a2*bc)*bc
        s = s.expand().trigsimp()
        print(s)
        print("----")
        print(latex(s))


        return

    def getWedge(self,x,y):
        """
        docstring for getWedge
        x,y:
            1d array with length 3
        """
        getMat = lambda i,j: x[i]*y[j] - x[j]*y[i]

        wedge = []
        wedge.append(getMat(1,2))
        wedge.append(-getMat(0,2))
        wedge.append(getMat(0,1))

        return wedge

    def getSecondForm(self,X,u,v,trig=True):
        """
        docstring for getSecondForm
        """
        Xu = self.diffVector(X,u)
        Xv = self.diffVector(X,v)
        Xuu = self.diffVector(Xu,u)
        Xuv = self.diffVector(Xu,v)
        Xvv = self.diffVector(Xv,v)

        # normal vector
        N = self.getWedge(Xu,Xv)

        dotProd = lambda X,Y: sum([x*y for x,y in zip(X,Y)]) 
        if trig:
            E = dotProd(Xu,Xu).trigsimp()
            F = dotProd(Xu,Xv).trigsimp()
            G = dotProd(Xv,Xv).trigsimp()
        else:
            E = dotProd(Xu,Xu).expand().factor()
            F = dotProd(Xu,Xv).expand().factor()
            G = dotProd(Xv,Xv).expand().factor()
        S = E*G-F**2 
        if trig:
            S = S.expand().trigsimp()
            s = sqrt(S).expand().trigsimp()
        else:
            S = S.simplify()
            s = sqrt(S).simplify()
        # print(Xu)
        # print(Xv)
        print("S:", latex(S),latex(s))

        if trig:
            e = (dotProd(N,Xuu)/s).trigsimp()
            f = (dotProd(N,Xuv)/s).trigsimp()
            g = (dotProd(N,Xvv)/s).trigsimp()
        else:
            e = (dotProd(N,Xuu)/s).expand().factor()
            f = (dotProd(N,Xuv)/s).expand().factor()
            g = (dotProd(N,Xvv)/s).expand().factor()

        K = (e*g-f**2)/S
        H = (e*G-2*f*F+g*E)/(2*S)

        if trig:
            K = K.expand().trigsimp()
            H = H.expand().trigsimp()
        else:
            K = K.expand().expand().factor()
            H = H.expand().expand().factor()

        print("-----------------------------------")
        print("-----------------------------------")
        print("-----------------------------------")
        # print("e",e)
        print("e & = & %s\\\\"%(latex(e)))
        print("f & = & %s\\\\"%(latex(f)))
        print("g & = & %s\\\\"%(latex(g)))
        print("E & = & %s\\\\"%(latex(E)))
        print("F & = & %s\\\\"%(latex(F)))
        print("G & = & %s\\\\"%(latex(G)))
        print("K & = & %s\\\\"%(latex(K)))
        print("H & = & %s"%(latex(H)))

        return K,H

    def secondForm(self):
        """
        docstring for secondForm
        """
        u,v,r,a = symbols("u v r a")
        # X = [r*sin(u)*cos(v),r*sin(u)*sin(v),r*cos(u)]
        # K,H = self.getSecondForm(X,u,v)

        # X = [(a+r*cos(u))*cos(v),(a+r*cos(u))*sin(v),r*sin(u)]
        # K,H = self.getSecondForm(X,u,v)

        X = [u-u**3/3+u*v**2,v-v**3/3+v*u**2,u**2-v**2]
        K,H = self.getSecondForm(X,u,v,trig=False)

        return

    def getXYSol(self,a,b,x1=1,y1=1,N=int(1e4)):
        """
        docstring for getXYSol
        a,b denotes x'(0) and y'(0)
        """
        dt = 1/N

        gamma121 = lambda x,y: 2*y/(1+y*y+x*x)
        gamma122 = lambda x,y: 2*x/(1+y*y+x*x)

        X  = [x1,x1+a*dt]
        Y  = [y1,y1+b*dt] 

        length = 0
        for i in range(N-1):

            dx = X[-1]-X[-2]
            dy = Y[-1]-Y[-2]
            dz = X[-1]*Y[-1] - X[-2]*Y[-2]
            length += (dx*dx+dy*dy+dz*dz)**0.5

            res = dx*dy
            x = gamma121(X[-2],Y[-2])*res 
            x = 2*X[-1] - X[-2] - x 
            y = gamma122(X[-2],Y[-2])*res
            y = 2*Y[-1] - Y[-2] - y
            X.append(x)
            Y.append(y)

        return X,Y,length

    def xyGeodesic(self):
        """
        docstring for xyGeodesic
        z = xy
        (1,1,1) -> (2,3,6)
        """

        getError = lambda a,b,c,d: ((a-b)**2+(c-d)**2)**0.5
         
        N = int(1e4)

        a1,b1 = 1.7,2.5
        delta = 0.02
        data = []
        p = 100
        n,m = 11,11
        for i in range(n):
            for j in range(m):
                a = a1 + i*delta
                b = b1 + j*delta
                X,Y,length = self.getXYSol(a,b,N=N)
                error = getError(X[-1],2,Y[-1],3)
                if error < p:
                    p = error
                    A,B = a,b
                    L  = length
                res = [a,b,X[-1],Y[-1],length,error]
                data.append(res)


        print("a,b,error and length: ",A,B,p,L)
        X,Y,length = self.getXYSol(A,B,N=N)
        print(X[-1],Y[-1])
        X = np.array(X)
        Y = np.array(Y)
        Z = X*Y
        # print(Z[-100:])
        T = np.arange(N+1)

        self.plot3D(X,Y,Z)

        
        return

    def plot3D(self,X,Y,Z,show=False):
        """
        docstring for plot3D
        """

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.plot(X,Y,Z, label='parametric curve')

        # plt.plot(T,X)
        # plt.plot(T,Y)
        if show:
            plt.show()

        return ax
    def getChrisSymbols(self,X,u,v,trig=True):
        """
        docstring for getChrisSymbols
        """
        Xu = self.diffVector(X,u)
        Xv = self.diffVector(X,v)

        dotProd = lambda X,Y: sum([x*y for x,y in zip(X,Y)]) 
        if trig:
            E = dotProd(Xu,Xu).trigsimp()
            F = dotProd(Xu,Xv).trigsimp()
            G = dotProd(Xv,Xv).trigsimp()
        else:
            E = dotProd(Xu,Xu).expand().factor()
            F = dotProd(Xu,Xv).expand().factor()
            G = dotProd(Xv,Xv).expand().factor()

        # Gamma 111,112,121,122,221,222
        Eu = E.diff(u)/2
        Ev = E.diff(v)/2
        Fu = F.diff(u)
        Fv = F.diff(v)
        Gu = G.diff(u)/2
        Gv = G.diff(v)/2
        s = E*G-F**2
        s = s.expand().simplify()
        efg = Matrix([[G,-F],[-F,E]])/s
        g11 = efg*Matrix([Eu,Fu-Ev])
        g12 = efg*Matrix([Ev,Gu])
        g22 = efg*Matrix([Fv-Gu,Gv])
        Gamma = []
        print(g11)
        Gamma.append(g11[0,0])
        Gamma.append(g11[1,0])
        Gamma.append(g12[0,0])
        Gamma.append(g12[1,0])
        Gamma.append(g22[0,0])
        Gamma.append(g22[1,0])
        # print(g11)
        # print(g12)
        # print(g22)

        return Gamma 

    def christoffel(self):
        """
        docstring for christoffel
        """
        u,v,r = symbols("u v r")
        a,b,c = 1,1,1
        X = [a*sin(u)*cos(v),b*sin(u)*sin(v),c*cos(u)]
        Gamma = self.getChrisSymbols(X,u,v)
        print(Gamma)

        X = [u,v,u*v]

        Gamma = self.getChrisSymbols(X,u,v)


        print(Gamma)
        return

    def getMullerPotential(self,x,y):
        """
        docstring for getMullerPotential
        x,y:
            float values
        """
        # parameters
        A  = [-200,-100,-170,15]
        a  = [-1,-1,-6.5,0.7]
        b  = [0,0,11,0.6]
        c  = [-10,-10,-6.5,0.7]
        x0 = [1,0,-0.5,-1]
        y0 = [0,0.5,1.5,1]

        # value of the function and the derivatives

        res = 0
        fx  = 0 
        fy  = 0
        fxx = 0
        fxy = 0
        fyy = 0

        for i in range(4):
            dx   = x - x0[i]
            dy   = y - y0[i]
            B    = a[i]*dx*dx+b[i]*dx*dy + c[i]*dy*dy
            res1 = A[i]*np.exp(B)
            res += res1
            cx   = 2*a[i]*dx + b[i]*dy
            cy   = 2*c[i]*dy + b[i]*dx
            fx  += cx*res1
            fy  += cy*res1
            fxx += (2*a[i]+cx*cx)*res1
            fxy += (b[i]+cx*cy)*res1
            fyy += (2*c[i]+cx*cx)*res1


        values = [res,fx,fy,fxx,fxy,fyy]

        return values

    def searchSol(self,A,n,C,func):
        """
        docstring for searchSol
        A:
            [a1,b1,delta,x1,y1]
        n:
            integer
        B:
            [x,y]
        func:
            function
        """
        getError = lambda a,b,c,d: ((a-b)**2+(c-d)**2)**0.5
         
        N = int(1e4)

        a1,b1,delta,x1,y1 = A
        p = 1000
        m = n
        A,B = 0,0
        for i in range(n):
            for j in range(m):
                print("i,j = %d,%d"%(i,j),A,B,p)
                a = a1 + i*delta
                b = b1 + j*delta
                X,Y,length = func(x1,y1,a,b)
                error = getError(X[-1],C[0],Y[-1],C[1])
                if error < p:
                    p = error
                    A,B = a,b
                    L  = length
                    X1 = X 
                    Y1 = Y
                    print("final point",X[-1],Y[-1])

        print("a,b,error and length: ",A,B,p,L)

        return A,B,L,X1,Y1

    def getMullerSol(self,x1,y1,a,b,N=int(1e4)):
        """
        docstring for getMullerSol
        a,b denotes x'(0) and y'(0)
        """
        dt = 1/N

        X  = [x1,x1+a*dt]
        Y  = [y1,y1+b*dt] 

        length = 0
        total = 10*N
        for i in range(total):

            dx = X[-1]-X[-2]
            dy = Y[-1]-Y[-2]
            dz = X[-1]*Y[-1] - X[-2]*Y[-2]
            length += (dx*dx+dy*dy+dz*dz)**0.5

            values = self.getMullerPotential(X[-2],Y[-2])
            fx,fy,fxx,fxy,fyy = values[1:]
            s = 1+fx*fx+fy*fy 
            g111 = fx*fxx/s
            g121 = fx*fxy/s
            g221 = fx*fyy/s
            g112 = fy*fxx/s
            g122 = fy*fxy/s
            g222 = fy*fyy/s
            x = g111*dx*dx+2*g121*dx*dy+g221*dy*dy
            x = 2*X[-1] - X[-2] - x 
            y = g112*dx*dx+2*g122*dx*dy+g222*dy*dy
            y = 2*Y[-1] - Y[-2] - y
            X.append(x)
            Y.append(y)

        return X,Y,length

    def mullerPotential(self):
        """
        docstring for mullerPotential
        """
        x,y = -0.1,1.5
        dt = 1e-4
        for i in range(10):
            values = self.getMullerPotential(x,y)
            x = x - dt*values[1]
            y = y - dt*values[2]

        print(x,y,i,values[:3])
        # print(self.getMullerPotential(-0.558,1.442)[:3])
        # print(self.getMullerPotential(0.623,0.028)[:3])

        print("geodesic of Muller surface")

        x1,y1 = -0.558,1.442
        a1,b1 = -5,-5
        delta = 2
        n = 0
        A = [a1,b1,delta,x1,y1]
        B = [0.623,0.028]
        # A,B,L,X,Y = self.searchSol(A,n,B,self.getMullerSol)
        # print(X[:100])
        a,b = 1,-1
        X,Y,L = self.getMullerSol(x1,y1,a,b,N=30000)
        print(X[-1],Y[-1],L)
        Z = []
        X1,Y1 = [],[]
        for i in range(0,len(X),1000):
            x,y = X[i],Y[i]
            # res = self.getMullerPotential(x,y)
            # Z.append(res[0])
            X1.append(x)
            Y1.append(y)
        # print(Z[-10:])
        # self.plot3D(X1, Y1, Z)
        plt.plot(X1,Y1)
        plt.show()

        return


    def mobiusStrip(self):
        """
        docstring for mobiusStrip
        """
        print("get the curvature of the mobius strip")
        u,v,r = symbols("u v r")
        X = [(1+v*cos(u/2)/2)*cos(u),
             (1+v*cos(u/2)/2)*sin(u),
             v*sin(u/2)/2]

        self.getSecondForm(X,u,v)

        return

    def plot3DSurface(self,X,Y,Z,show=False):
        """
        docstring for plot3DSurface
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)

        if show:
            plt.show()


        return ax

    def mullerSurface(self):
        """
        docstring for mullerSurface
        """
        X = np.arange(-0.6, 0.7, 0.05)
        Y = np.arange(-0.1, 1.5, 0.05)
        n,m = len(X),len(Y)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros((m,n))
        # print(X.shape,Y.shape,n,m)
        # print(X[:5,:5])
        for i in range(m):
            for j in range(n):
                print(i,j)
                Z[i,j] = self.getMullerPotential(X[i,j],Y[i,j])[0]

        ax = self.plot3DSurface(X, Y, Z)
        x1,y1,a,b = 0.6,0.03,-0.2,0.2
        X,Y,L = self.getMullerSol(x1,y1,a,b,N=10000)
        Z = []
        for i,j in zip(X,Y):
            Z.append(self.getMullerPotential(i,j)[0])
        # self.plot3D(X, Y, Z)
        ax.plot(X,Y,Z, label='parametric curve')

        plt.show()


        return

    def checkSign(self,eight,sign,j=0):
        """
        docstring for checkSign
        """
        a,b,c,d,e,f,g = sign
        signs = [[1,1,1,1,1,1,1,1],
                 [-1,1,a,-a,b,-b,c,-c],
                 [-1,-a,1,a,d,e,-d,-e],
                 [-1,a,-a,1,f,g,-g,-f],
                 [-1,-b,-d,-f,1,b,d,f],
                 [-1,b,-e,-g,-b,1,g,e],
                 [-1,-c,d,g,-d,-g,1,c],
                 [-1,c,e,f,-f,-e,-c,1]]
        signs = Matrix(signs)
        self.eight = Matrix.zeros(8)
        for i in range(8):
            for j in range(8):
                self.eight[i,j] = eight[i,j]*signs[i,j]
        number = 0 
        for i in range(8):
            res = 0 
            for k in range(8):
                res += eight[j,k]*eight[i,k]*signs[i,k]*signs[j,k]
            if res == 0:
                number += 1
        if number == 7:
            return True 
        else:
            return False

    def eightSquares(self):
        """
        docstring for eightSquares
        """

        x = symbols("x0:9")
        eight = [[x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]],
                 [x[2],x[1],x[4],x[3],x[6],x[5],x[8],x[7]],
                 [x[3],x[4],x[1],x[2],x[7],x[8],x[5],x[6]],
                 [x[4],x[3],x[2],x[1],x[8],x[7],x[6],x[5]],
                 [x[5],x[6],x[7],x[8],x[1],x[2],x[3],x[4]],
                 [x[6],x[5],x[8],x[7],x[2],x[1],x[4],x[3]],
                 [x[7],x[8],x[5],x[6],x[3],x[4],x[1],x[2]],
                 [x[8],x[7],x[6],x[5],x[4],x[3],x[2],x[1]]]
        eight = Matrix(eight)
        sol1 = []
        combi = itertools.product([-1,1],repeat=7)
        for line in combi:
            sol1.append(list(line))
        # print(sol1)
        for i,sign in enumerate(sol1):
            count = 0
            for j in range(8):
                if self.checkSign(eight,sign,j=j):
                    count += 1 
                else:
                    break 
            if count == 8:
                print(i,sign)
            break

        sign = [1, 1, 1, 1, -1, -1, -1] 
        self.checkSign(eight,sign,j=0)
        print(self.eight)
        for i in range(8):
            for j in range(i+1,8):
                res = 0 
                for k in range(8):
                    res += self.eight[i,k]*self.eight[j,k]
                print(i,j,res)
        print(latex(self.eight))

        return 

    def checkOmega(self,omega):
        """
        docstring for checkOmega
        omega:
            1d array, such as [1,1,1,1,1]
        return:
            integer or list
            [a,b,b,b,b,...] => a-b
            or => omega 
        """

        for i in omega[2:]:
            if i != omega[1]:
                return omega 
        num = omega[0] - omega[1]

        return num 

    def getPolyRootCoef(self,Xu,k):
        """
        docstring for getPolyRootCoef
        relations between the coefficients and the root
        Xu:
            1d array, [X,u1,u2]
        k:
            integer between 1 and len(X)
        """
        X,u1,u2,w = Xu
        n = len(X)
        s = 0
        combi = itertools.combinations(np.arange(n),k)
        for line in combi:
            res = 1
            for i in line:
                res = res*X[i]
            s += res 
        res = s.expand().collect([u1,u2])
        res = Poly(res,[u1,u2]).as_dict()
        for key in res:
            dicts = {}
            for i in range(n):
                dicts[i] = 0 
            s = res[key]
            s = Poly(s,w).as_dict()
            for key0 in s:
                dicts[key0[0]%n] += s[key0]
            num = self.checkOmega(list(dicts.values()))
            if num != 0:
                print(key,num)

        return  
    def deMoivreQuintic(self):
        """
        docstring for deMoivreQuintic
        0&=&x^{5}+5ax^{3}+5a^{2}x+b
        """
        a = 1
        b = 1 
        w = np.exp(2*np.pi*1j/5)
        m = (b*b + 4*a**5)**0.5 
        u1 = ((-b+m)/2)**(1/5)
        u2 = -((b+m)/2)**(1/5)
        print(u1,u2)

        X = []
        W = []
        for i in range(5):
            W.append(w**i)
        for i in range(5):
            x = u1*W[i] + u2*W[(4*i)%5]
            X.append(x)
            print(i,x,x**5+5*a*x**3+5*a**2*x+b)

        return 

    def highOrderEqn(self):
        """
        docstring for highOrderEqn
        """
        w,u1,u2 = symbols("w u1 u2")
        # w = exp(2*pi*I/5)
        n = 13
        X = []
        W = []
        for i in range(n):
            W.append(w**i)
        for i in range(n):
            X.append(u1*W[i] + u2*W[(-i)%n])
         
        Xu = [X,u1,u2,w]
        for k in range(1,n+1):
            print("------ k = %d ------"%(k))
            self.getPolyRootCoef(Xu,k)
        return

    def highOrderEqnByMat(self):
        """
        docstring for highOrderEqnByMat
        """
        p = Integer(13)

        S = []
        n = (p-1)//2 
        p = Symbol("p")
        for i in range(1,n+1):
            num = factorial(2*i)/(factorial(i))**2
            S.append(p*num)

        print(S)

        A = []
        A.append(-S[0]/2)

        for i in range(1,n):
            res = 0 
            for j in range(i):
                res += S[i-j-1]*A[j]

            num = (-S[i]-res)/(2*i+2)
            A.append(num)
            print(i,num.factor())
        # print(A)

            
        return
    def test(self):
        """
        docstring for test
        """

        # self.eightSquares()
        # self.deMoivreQuintic()
        # self.highOrderEqn()
        self.highOrderEqnByMat()


        return

puzzle = Puzzles()
puzzle.test()

