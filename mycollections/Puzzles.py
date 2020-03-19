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
        print(latex(A))
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
        p,q = 11,44
        s = x**5 + p*x + q 
        print(s.factor())

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

    def test(self):
        """
        docstring for test
        """
        # self.testRamanujanPi1()
        # self.alibabaPuzzles()
        # self.testGalois()
        self.testEqnDet()
        # self.testCharacters()

        return

puzzle = Puzzles()
puzzle.test()


        
