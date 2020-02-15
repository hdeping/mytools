#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-12-14 10:50:13
    @project      : Formulas Deriations
    @version      : 1.0
    @source file  : Formulas.py

============================
"""

import sympy
from sympy import expand,simplify,cos,sin,exp,sqrt
from sympy import latex,Symbol,diff,solve,factor
from sympy import sympify,trigsimp,expand_trig
from sympy import Matrix,limit,tan,Integer,symbols,Poly
from sympy.solvers import diophantine
import numpy as np
from mytools import MyCommon
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import json
import time
import itertools
from decimal import Decimal
import re


class EllipticCurve():
    """
    formula deriation for the elliptic curve 
    with sympy
    """
    def __init__(self):
        super(EllipticCurve, self).__init__()
    def formula(self):
        """
        docstring for formula
        elliptic curve l:
            y^{2}&=&ax^{3}+bx^{2}+cx+d
        A,B:
            A,B&=&\left(x_{1},y_{1}\right),\left(x_{2},y_{2}\right)
        C is counterpart of the third intersection point of line AB and l
        """
        # point A and B
        x1 = sympy.Symbol("x1")
        x2 = sympy.Symbol("x2")
        y1 = sympy.Symbol("y1")
        y2 = sympy.Symbol("y2")
        a  = sympy.Symbol("a")
        b  = sympy.Symbol("b")
        c  = sympy.Symbol("c")
        d  = sympy.Symbol("d")
        x  = sympy.Symbol("x")
        k  = sympy.Symbol("k")
        # line AB
        # k  = a*(x1**2+x1*x2+x2**2)+b*(x1+y2)+c
        # k  = k/(y1+y2)
        y  = k*(x-x1)+y1
        print("k: ",k)
        print("y: ",y)
        print(sympy.factor(y))
        s = a*x**3+b*x**2+c*x+d - y**2 
        s = sympy.expand(s)
        s = sympy.collect(s,x)
        print(s)
        s = sympy.latex(s)
        print(s)
        x3 = (k**2 - b)/a - x2 - x1
        y3 = - k*(x3 - x1) - y1
        print("C: (x3,y3)")
        print(x3,y3)

        x = sympy.sqrt(3)
        y = sympy.sqrt(2)
        x = (x-y)**3
        x = sympy.expand(x)
        print(x)
        
        
        return
    def func(self,p,x):
        """
        docstring for func
        input:
            p, [a,b,c,d], array with length 4
            x, input number
        """
        y = 0 
        for i in range(4):
            y = y*x+p[i]
        y = sympy.sqrt(y)
        return y
    def getThirdX(self,x1,x2=None,p=None):
        """
        docstring for getThirdX
        input:
            x1, input 1
            x2, input 2
        return:
            (x3,y3), coordinate of the 
            third intersection point
        """
        a,b,c,d = p 

        # y1 = self.func(p,x1)
        # k  = a*(x1**2+x1*x2+x2**2)+b*(x1+y2)+c
        # k  = k/(y1+y2)
        if x2 == None:
            x2 = x1
            k  = (3*a*x1[0]**2+2*b*x1[0]+c)/(2*x1[1])
            # k  = (3*a*x1*x1+2*b*x1+c)/(2*y1)
        else:
            # y2 = self.func(p,x2)
            # k  = (y2 - y1)/(x2 - x1)
            k  = (x2[1] - x1[1])/(x2[0] - x1[0])
        k  = sympy.simplify(k)
        x3 = (k*k - b)/a - x2[0] - x1[0]
        x3 = sympy.simplify(x3)
        # y3 = - k*(x3 - x1) - y1
        y3 = - k*(x3 - x1[0]) - x1[1]
        y3 = sympy.simplify(y3)
        # print("k",k,x3,y3)
        # print("k : ",k)
        # print("x3: ",x3)
        # print("y3: ",y3)
        return x3,y3
    def getCoPrimeList(self,n):
        """
        docstring for getCoPrimeList
        n:
            positive integer
        return:
            4 -> [1,3], number list which is less than n
            and coprime with n
            10 -> [1,3,7,9]
        """
        res = []
        divisors = sympy.factorint(n)
        for i in range(2,n):
            count = 0
            for j in divisors:
                if i%j == 0:
                    count = 1
                    break
            if count == 0:  
                res.append(i)
        
        return res
    def ellipticSol(self,k=4):
        """
        docstring for ellipticSol
        """
        # if k in [8]:
        #     return
        print("--------------- %d ---------------"%(k))
        p = [1,4*k*k+12*k-3,32*(k+3),0]
        # print(self.getThirdX([4,4*(2*k+5)],
        #                      x2=[8*(k+3),8*(k+3)*(2*k+5)],
        #                      p=p))
        print("p = ",p)
        # print(self.func(p,4))
        
        num = p[1]
        solutions = []
        for k in range(2,1000):
            if k%10 == 0:
                print(k)
            k = k*k
            primeList = self.getCoPrimeList(k)
            for i in range(-num,0):
                for coprime in primeList:
                    x = i + Integer(coprime)/k
                    y = self.func(p,x)
                    if y.is_rational:
                        print("(%d,%d/%d),"%(i,i*k+coprime,k))
                        solutions.append([i,i*k+coprime,k])
            print(k,x,y)

        if len(solutions) == 0:
            print("no points within %d"%(num))
            return
        else:
            print("there are %d solutions to equation"%(len(solutions)))
            return


       
        # self.getInverseTran()

            
        return

    def testEllipticPoints(self):
        """
        docstring for testEllipticPoints
        """
        
        # x1 = [Integer(solutions[0][0]),Integer(solutions[0][1])]
        # self.checkSolution(x1,p)
        # x1 = [Integer(2),Integer(3)]
        # x1 = [Integer(1),Integer(0)]
        x2 = [Integer(1152),Integer(111744)]
        x1 = [Integer(-6912),Integer(6912)]
        # p  = [1,0,0,1]
        p  = [1,7105,1327104,0]
        self.getEllipticPoints(x1,p)
        x3 = self.getThirdX(x1,x2=x2,p=p)
        print(x3)
        print(self.func(p,-1176))
        for i in range(4):
            x3 = self.getThirdX(x1,x2=x3,p=p)
            print(i,x3)
        return
    def getEllipticPoints(self,x1,p):
        """
        docstring for getEllipticPoints
        """
        print(0,x1)
        x3 = self.getThirdX(x1,p=p)
        print(1,x3)
        for i in range(6):
            x3 = self.getThirdX(x1,x2=x3,p=p)
            print(i+2,x3)
        return
    def checkSolution(self,x1,p):
        """
        docstring for checkSolution
        """
        x3 = self.getThirdX(x1,x2=None,p=p)
        print("2P",x3)
        for i in range(1000):
            # print(i,x3)
            # print(i)
            x3 = self.getThirdX(x1,x2=x3,p=p)
            s1 = x3[0]
            a  = x3[0]-x3[1]-8*(k+3)
            b  = x3[0]+x3[1]-8*(k+3)
            c  = (2*k+4)*x3[0] + 8*(k+3)
            p1 = (s1.is_nonnegative and a.is_nonnegative and b.is_nonnegative)
            p1 = (p1 and c.is_nonnegative)
            p2 = (s1.is_negative and a.is_negative and b.is_negative)
            p2 = (p2 and c.is_negative)
            # print(x3)

            if p1 or p2:
            # if i < 3:
                print("solution")
                m,n = sympy.fraction(a)
                a = -a*n
                b = -b*n
                c = -c*n
                print("k = ",k,i,"length: ",len(str(a)))
                # print("a = ",a)
                # print("b = ",b)
                # print("c = ",c)
                break
        # print(a,b,c)
        # a = 375326521
        # b = -679733219
        # c = -8106964*109
        # x3 = [4,28]
        # a  = x3[0]-x3[1]-8*(k+3)
        # b  = x3[0]+x3[1]-8*(k+3)
        # c  = (2*k+4)*x3[0] + 8*(k+3)


        print("check: ",a/(b+c)+b/(a+c)+c/(a+b))
        return
    def getInverseTran(self):
        """
        docstring for getInverseTran
        """
        k = Symbol("k")
        b = 4*k*k+12*k-3
        c = 32*(k+3)
        s = expand(b*b-4*c)
        s = factor(s)
        print(latex(s))
        print(solve(s,k))
        x = 8*(k+3)
        s = sqrt(factor(x**3+b*x**2+c*x))
        s = sympy.powsimp(s)
        print(s)

        print(factor(3*x**2+2*b*x+c))

        M = Matrix([[1,-1,-8*(k+3)],
                    [1,1,-8*(k+3)],
                    [2*k+4,0,8*(k+3)]])
        X = sympy.symbols("x y z")
        A = sympy.symbols("a b c")
        X = Matrix(X)
        A = Matrix(A)
        sol = M*X-A
        sol = solve(sol,X)
        print()
        x = factor(sol[X[0]])
        y = factor(sol[X[1]])
        z = factor(sol[X[2]])
        print(latex(x))
        print(latex(y))
        print(latex(z))
        return

    def ellipticSol2(self):
        """
        docstring for ellipticSol
        """
        a = Symbol("a")
        b = Symbol("b")
        c = Symbol("c")
        # k = Symbol("k")
        k = 4
        s = a+b+c 
        y = (a/(s-a)+b/(s-b)+c/(s-c)-k)*(s-a)*(s-b)*(s-c)
        y = simplify(y)
        y = expand(y)
        y = sympy.collect(y,k)
        target = y
        print(latex(y))

        x = self.xyz[0]
        y = self.xyz[1]
        target = target.subs(a,x-y-56)
        target = target.subs(b,x+y-56)
        target = target.subs(c,4*(3*x+14))
        target = expand(target)
        target  = factor(target)
        print(latex(target))

        
        # self.getTargetForm()
        
        t  = sympy.symbols("t1:3")
        s1 = -28*(1+t[0]+2*t[1]) - x*(6+6*t[0] - t[1])
        s2 = 364*(1-t[0]) - y*(6+6*t[0] - t[1])
        sol = solve([s1,s2],t)
        print(latex(sol[t[0]]))
        print(latex(sol[t[1]]))

        # a = x-y-56
        # b = x+y-56
        # c = 4*(3*x+14)
        # coef = sympy.symbols("c0:5")
        b = sympy.symbols("b0:2")
        k = Symbol("k")
        # k = 4
        coef = [0,1,b[0],2*k+4,-b[0]]
        a = x-coef[1]*y+coef[2]
        b = x+coef[1]*y+coef[2]
        c = coef[3]*x+coef[4]
        s1 = expand(a+b+c)
        s2 = expand(a**2+b**2+c**2)
        s3 = expand(a**3+b**3+c**3)
        s12 = expand(s1*s2)
        abc = expand(a*b*c)
        print(latex(s1) + "\\\\")
        print(latex(s2) + "\\\\")
        print(latex(s12) + "\\\\")
        print(latex(s3) + "\\\\")
        print(latex(abc)+ "\\\\")

        
        total = expand(k*s3-(k-1)*s12-(2*k-3)*abc)
        total = simplify(total)
        total = total.collect([x,y])
        print(latex(total))

        print(total)

        p = [1,109,224,0]
        print(self.func(p,4))
        x1 = [Integer(4),Integer(52)]
        x3 = self.getThirdX(x1,x2=None,p=p)
        print(x3)
        for i in range(1):
            x3 = self.getThirdX(x1,x2=x3,p=p)
            print(i,x3,x3[0]-x3[1]-56,x3[0]+x3[1]-56)
        print(factor(16*k*k+88*k+120))
        b = sympy.symbols("b0:2")
        b0 = b[0]
        b1 = -b[0]
        s  = -2*b0**3*k + 4*b0**3 - 4*b0**2*b1*k
        s += 5*b0**2*b1 - 2*b0*b1**2*k + 2*b0*b1**2 + b1**3
        s = factor(s)
        print(latex(s))
        s   = -8*b0*k**3 - 40*b0*k**2 - 18*b0*k + 84*b0
        s  += 4*b1*k**2 + 36*b1*k + 69*b1
        s  = factor(s)
        # s = s.collect(k)
        print(latex(s))
        s  = -8*b0**2*k**2 - 12*b0**2*k + 32*b0**2 
        s += 4*b1**2*k + 14*b1**2
        s += - 8*b0*b1*k**2 - 16*b0*b1*k + 26*b0*b1 
        s = factor(s)
        # s = s.collect(k)
        print(latex(s))

       


        return
    def getTargetForm(self):
        """
        docstring for getTargetForm
        """
        k = Symbol("k")
        t = sympy.symbols("t1:3")
        s = 1 + t[1] + t[0]
        y = (1/(s-1)+1/(s-t[0])+1/(s-t[1])-k)*(s-1)*(s-t[0])*(s-t[1])
        y = simplify(y)
        y = expand(y)
        y = sympy.collect(y,k)
        target = y
        print(latex(y))

        # replace t1 and t2 with x and y 
        x = self.xyz[0]
        y = self.xyz[1]
        c = sympy.symbols("c1:10")
        print(c)
        s1 = c[3]*x+c[4]*y+c[5] - t[0]*(c[0]*x+c[1]*y+c[2])
        s2 = c[6]*x+c[7]*y+c[8] - t[1]*(c[0]*x+c[1]*y+c[2])
        sol = solve([s1,s2],[x,y])

        x = expand(sol[x])
        y = expand(sol[y])
        x = simplify(x)
        y = simplify(y)
        print(latex(x),latex(y))
        
        x = self.xyz[0]
        y = self.xyz[1]
        # s1 = (c[3]*x+c[4]*y+c[5])/(c[0]*x+c[1]*y+c[2])
        # s2 = (c[6]*x+c[7]*y+c[8])/(c[0]*x+c[1]*y+c[2])
        # target = target.subs(t[0],s1)
        # target = target.subs(t[1],s2)

        a = c[0]*x+c[1]*y+c[2]
        b = c[3]*x+c[4]*y+c[5]
        c = c[6]*x+c[7]*y+c[8]
        s1 = a+b+c 
        s2 = a**2+b**2+c**2 
        s3 = a**3+b**3+c**3 
        # s1 = expand(s1).collect([x,y])
        # s2 = expand(s2).collect([x,y])
        # s3 = expand(s3).collect([x,y])
        # abc = expand(a*b*c).collect([x,y])
        # print("s1",latex(s1),"\\\\")
        # print("s2",latex(s2),"\\\\")
        # print("s3",latex(s3),"\\\\")
        # print("abc",latex(abc),"\\\\")
        # target = (a/(s-a)+b/(s-b)+c/(s-c)-k)*(s-a)*(s-b)*(s-c)
        # target = (a)*(s-b)*(s-c)+(s-a)*(b)*(s-c)+(s-a)*(s-b)*(c)
        # target += -k*(s-a)*(s-b)*(s-c)
        # target = simplify(target)
        # target = expand(target)
        print(target)
        # print("target",latex(target))
        x = self.xyz[0]
        # x = solve(x**4-12*x**3+16*x+24)
        # print(latex(x))
        y = sympy.symbols("y1:3")
        x  = [3+y[0]+y[1]]
        x.append(3+y[0]-y[1])
        x.append(3-y[0]+y[1])
        x.append(3-y[0]-y[1])
        s = 0
        for i in range(4):
            for j in range(i+1,4):
                s += x[i]*x[j]
        s = expand(s)
        print(s)
        s = expand(x[0]*x[1]*(x[2]+x[3])+x[2]*x[3]*(x[0]+x[1]))
        print(s)
        s = expand(x[0]*x[1]*(x[2]*x[3]))
        print(s)

        x = Integer(4)**(Integer(1)/3) 
        x += 3*Integer(2)**(Integer(1)/3)
        x += 9
        # print(x)
        x = self.xyz[0]
        y = expand(2*x**3-(3*x-25)**3)
        y = factor(y)
        print(y)
        
        


        return

    def testNagellLutz(self):
        """
        docstring for testNagellLutz
        Nagell-Lutz's theorem for the torsion subgroup 
        of the elliptic curve
        """
        k = Symbol("k")
        k = 3
        p = [1,4*k*k+12*k-3,32*(k+3),0]
        print(p)
        x = self.xyz[0]
        y = self.func(p,x-p[1]/Integer(3))
        y = expand(y).collect(x)
        print(y)
        A = -1395
        B0 = 19918
        


        # k = 3, delta = 2^12 * 3^3 * 11^3
        num = np.array([12,3,3]) // 2  + 1
        primes = [2,3,11]
        # print(num)
        for i in range(num[0]):
            for j in range(num[1]):
                for k in range(num[2]):
                    y = 1 
                    y *= primes[0]**i
                    y *= primes[1]**j
                    y *= primes[2]**k
                    B  = B0 - y**2
                    exp = x**3 + A*x + B 
                    exp = factor(exp)
                    print(y,exp)

        p = [1,0,A,B0]
        print(self.func(p,27))
        print(self.func(p,71))
        k = 3
        p = [1,4*k*k+12*k-3,32*(k+3),0]
        print(self.func(p,4))
        print(self.func(p,48))
        return
    def getDelta(self):
        """
        docstring for getSolDelta
        """
        B = 128*k**6/27 + 128*k**5/3 + 352*k**4/3 + 64*k**3/3 - 344*k**2 - 328*k + 94 
        A = -16*k**4/3 - 32*k**3 - 40*k**2 + 56*k + 93
        # B = B*27
        # A = A*3 
        print("B",latex(B))
        print("A",latex(A))
        # B = factor(B)
        # A = factor(A)
        print("B",latex(B))
        print("A",latex(A))
        delta = 4*A**3 + 27*B**2
        delta = factor(expand(delta))
        print("delta",latex(delta))

        return
    def testElliptic(self):
        """
        docstring for testElliptic
        """
        # p  = [1,0,-2,2]
        # x1 = [0,sqrt(2)]
        # x3 = self.getThirdX(x1,p=p)
        # print(x3)
        # print(p)
        # print(self.func(p,Integer(1)/2))
        # for i in range(3):
        #     x3 = self.getThirdX(x1,x2=x3,p=p)
        #     print(i,x3)
        # print(self.func(p,x3[0]))
        print(self.getCoPrimeList(30))
        for k in range(8,9,2):
            self.ellipticSol(k=k)
        # self.getInverseTran()
        # self.testNagellLutz()

        # for k in range(1,1000):
        #     p = [1,4*k*(k+3)-3,32*(k+3)]
        #     a = sympy.factorint(p[1])
        #     b = sympy.factorint(p[1]**2-4*p[0]*p[2])
        #     rank = len(a) + len(b) - 1 
        #     print(k,rank,a,b)
        return
    

class Formulas(MyCommon,EllipticCurve):
    """
    Formulas Deriations with sympy
    self.one:
    self.zero:
        one and zero in sympy
    self.xyz: 
        x, y and z
    """
    def __init__(self):
        super(Formulas, self).__init__()

        self.one  = sympy.ones(1)[0,0]
        self.zero = sympy.zeros(1)[0,0]
        self.xyz  = []
        self.xyz.append(sympy.Symbol("x"))
        self.xyz.append(sympy.Symbol("y"))
        self.xyz.append(sympy.Symbol("z"))

        return 
    def run(self):
        """
        docstring for run
        """
        x = []
        for i in range(1,6):
            symbol = sympy.Symbol("x%d"%(i))
            x.append(symbol)

        summation = 0
        for i in range(3):
            for j in range(i+1,4):
                for k in range(j+1,5):
                
                    summation += x[i]*x[j]*x[k]
        print(summation)
        s = sum(x)*summation
        s = sympy.expand(s)
        print(s)
    
        return
    def getBinomial(self,n):
        """
        docstring for getBinomial
        n:
            positive integer and n >= 2
        return:
            [1,n,C_n^i...]
        """
        assert(n >= 2)
        binomial = [1]
        for i in range(n-2):
            num = binomial[i]*(n-i) // (i+1)
            binomial.append(num)

        return binomial
    def getExpSum(self,s,n,m):
        """
        docstring for getExpSum
        s:
            [s0,s1...,s_{m-1}]
        n:
            symbol "n"
        m:
            positive number, m >= 2
        return:
            P_m(n)
            P_{m}(n)&=&\sum_{k=1}^{n}k^{m}
        """
        result   = (n+1)**(m+1) - 1 
        binomial = self.getBinomial(m+1)
        for i in range(m):
            result = result - binomial[i]*s[i]
            
        result = result / (m+1)
        result = sympy.expand(result)
        # result = sympy.factor(result)
        return result

    def bernoulliNum(self):
        """
        docstring for bernoulliNum
        """
        n = sympy.Symbol("n")
        s = [n,n*(n+1)/2]

        for m in range(2,9):
            Pm = self.getExpSum(s,n,m)
            print(sympy.latex(Pm))
            s.append(Pm)

        return

    def bernoulliGen(self):
        """
        docstring for bernoulliGen
        """
        x = sympy.Symbol("x")
        s = x/(sympy.exp(x) - 1)
        # for i in range(30):
        #     s = sympy.diff(s,x,2)
        #     value = sympy.limit(s,x,0)
        #     print("B%d = "%(i*2+2),sympy.latex(value))
        s = sympy.limit(s,x,0)
        print((s/234)**2)
        
        
        return
    def getBernoulli(self,s,m):
        """
        docstring for getBernoulli
        """
        binomial = self.getBinomial(m+1)
        result   = s[0]
        for i in range(m):
            result = result - binomial[i]*s[i]/(m+1)

        return result
    def bernoulliNum2(self):
        """
        docstring for bernoulliNum2
        """
        x = sympy.Symbol("x")
        zero = sympy.diff(x,x,2)
        s = [sympy.diff(x,x)]
        s.append(s[0]/2)

        for i in range(2,102,2):
            result = self.getBernoulli(s,i)
            s.append(result)
            s.append(zero)
            print("B_{%d} & = & "%(i),sympy.latex(result), "\\\\")
            


        return
    def getDivisorSeq(self,m,n):
        """
        docstring for getDivisorSeq
        m,n:
            two positive integers
        return:
            divisor, a inversed array
        for example, (11,3) => [1,3]
        """
        assert(m > 0 and n > 0)
        divisor = []
        while n != 1:
            divisor.append(m//n)
            k = n 
            n = m%n
            m = k

        divisor = np.array(divisor)
        divisor = np.flip(divisor)
        return divisor

    def getModuloSeq(self,m,n):
        """
        docstring for getModuloSeq
        m,n:
            two positive integers
        return:
            divisor, a inversed array
        for example, (19,11) => [19,11,8,3,2]
        """
        assert(m > 0 and n > 0)
        divisor = []
        while n != 1:
            divisor.append(m)
            k = n 
            n = m%n
            m = k
        divisor.append(m)
        return divisor
    def diophantine(self,m,n):
        """
        solution to diophantine equation
        \begin{eqnarray*}
        x_{1}&=&1\\
        x_{2}&=&-b_{1}\\
        x_{k+1}&=&-b_{k}x_{k}+x_{k-1}\\
        (x,y)&=&\left(x_{n-1},x_{n}\right)
        \end{eqnarray*}
        m,n:
            two integers which is coprime to each other
        
        """
        

        divisor = self.getDivisorSeq(m,n)
        solution = [1,-divisor[0]]
        for i in range(1,len(divisor)):
            value = - divisor[i]*solution[i] + solution[i-1]
            solution.append(value)

        # print(solution)
        print("solution: ",solution[-2:])
        print("%d*%d+%d*%d = 1"%(m,solution[-2],n,solution[-1]))
        print(m*solution[-2]+n*solution[-1])


        return solution[-2:]
    def remainderTheorem(self):
        """
        docstring for remainderTheorem
        check the remainder theorem based 
        on the diophantine equation
        """
        p = np.array([3,5,7])
        P = np.prod(p)
        q = P // p
        a = [2,3,2]

        solution = 0
        for i in range(len(p)):
            value = self.diophantine(p[i],q[i])[1]
            print(a[i],value)
            solution += value*a[i]*q[i]
        print(solution)
        solution = solution % P
        print(solution)
        print(solution % p)
        

        return
    def pellSol(self,D,n):
        """
        docstring for pellSol
        x_{2}   &=& x_{1}^{2}+Dy_{1}^{2}\\
        x_{n+1} &=& 2x_{1}x_{n}-x_{n-1}\\
        y_{2}   &=& 2x_{1}y_{1}\\
        y_{n+1} &=& 2x_{1}y_{n}-y_{n-1}
        D:
            positive integer
        n: 
            positive integer
        """
        a,b = self.getInitPell(D)
        x = [[1,0],[a,b]] # D = 2
        
        for i in range(n):
            x0 = 2*a*x[-1][0] - x[-2][0]
            x1 = 2*a*x[-1][1] - x[-2][1]
            # print("|%d|%d|%d|%d|"%(i+2,x0,x1,x0**2 - D*x1**2))
            x.append([x0,x1])
        
        return x
    def getInitPell1(self,D):
        """
        docstring for getInitPell
        D:
            x^2 - Dy^2 = 1, 
            positive integer and not square one and  D > 1
            2m+1, D^2 = (m+1)^2 - m^2
            2m, D^2 = (m^2+1)^2 - (m^2-1)^2
            x^2 = D
            x^2 - p^2 = q^2
            x = p + q/(x+p)
        """
        m = D // 2
        if D % 2 == 1:
            p = m + 1
            q = m 
        else:
            p = m**2 + 1
            q = m**2 - 1 

        one = sympy.ones(1)[0,0]
        zero = sympy.zeros(1)[0,0]
        x   = 0
        for i in range(10):
            x = p + q/(p+x)
            print(i+1,x,x**2 - D)
        
        return  

    def getContinueSeq(self,D,n=2,target=1,count=100):
        """
        docstring for getInitPell
        D:
            x^2 - Dy^2 = 1, 
            positive integer and not square one and  D > 1
            x_n = a_n + 1/(a_{n+1} + x_{n-1})
        """
        if n == 2:
            x = sympy.sqrt(D)
        else:
            x = D**(Integer(1)/n)
        # print(x)
        result = []
        if target == 1:
            target = 2*(x//1)

        for i in range(count):
            # print(i,latex(x))
            a = x // 1 
            x = sympy.simplify(1/(x - a))
            result.append(a)
            if a == target:
                break 

        return result

    def getCubicContinueSeq(self,D,n=3,count = 100):
        """
        docstring for getInitPell
        continued fraction of \sqrt[3]{D}
        """
    
        a = Decimal(D**(1/n))
        result = []
        p = [0,1]
        q = [1,0]
        for i in range(count):
            # x = (p[-2]+q[-2]*a)/(p[-1]+q[-1]*a)
            x  = Decimal(p[-1]**3 + D*q[-1]**3)
            y1 = Decimal(p[-2]**3 + D*q[-2]**3)
            b  = q[-1]*a
            c  = q[-2]*a
            y2 = Decimal(p[-1]**2 - b*p[-1] + b**2)
            y3 = Decimal(p[-2]**2 - c*p[-2] + c**2)
            x = y1*y2/y3/x 
            x = int(x)
            result.append(x)
            num = -x*p[-1] + p[-2]
            p.append(num)
            num = -x*q[-1] + q[-2]
            q.append(num)          

        return result 

    def getInitPell(self,D,num=1):
        """
        docstring for getInitPell
        D:
            x^2 - Dy^2 = num,
        """
        result = self.getContinueSeq(D)
        # print("result",result)
        m      = result[0]
        sequence = result[1:]
        A = [1,0]
        B = [0,1]
        n = 100
        for i in range(n):
            item = sequence[i % len(sequence)]
            a = item*A[-1] + A[-2]
            b = item*B[-1] + B[-2]
            x = m*b + a
            y = b 
            judge = x**2 - D*y**2
            A.append(a)
            B.append(b)
            # print(i+1,x,y,judge)
            # print(i,x,y,judge)
            if judge == num:
                # print(i+1,x,y,judge)
                return x,y
        print("there is no integer solutions within %d iterations"%(n))
        return None


    def testPell(self):
        """
        docstring for testPell
        """
        output = {}
        for i in range(2,1000): 
            if i == (int(i**0.5))**2:
                continue
            result = self.getInitPell(i) 
            res = {}
            res["number"] = i
            res["length"] = len(result) - 1 
            res["result"] = result
            output[str(i)] = res
            print(i,len(result)-1,result)

        self.writeJson("continuedFraction.json",output)
        
        return
    def dealData(self):
        """
        docstring for dealData
        read data from continued.json
        """
        data = self.loadStrings("continued.json")
        # print(data)
        x = []
        y = []
        stati = {}
        z = []
        output = {}
        for line in data:
            line  = line.split(" ")

            index = line[0]
            num   = line[1]
            res = {}
            res
            res["number"] = int(index)
            res["length"] = int(num)
            arr = np.array(line[2:])
            arr = arr.astype(int)
            res["result"] = arr.tolist()
            output[index] = res
            
            if num in stati:
                stati[num] += 1 
            else:
                stati[num] = 1
            if num == "27":
                print(line[:2])
                z.append(int(index))
        # plt.plot(x,y,"o")
        # plt.show()
        x = np.arange(len(z))
        # plt.plot(x,z,"o")
        # plt.show()
        # print(json.dumps(stati,indent = 4))
        res = []
        for key in stati:
            value = stati[key]
            res.append([int(key),value])

        res.sort()
        res = np.array(res)
        print(res)

        plt.bar(res[:,0],res[:,1])
        plt.savefig("continuedFraction.png",dpi=200)
        plt.show()


        # self.writeJson(output,"continuedFraction.json")

        return
    def continueFrac(self):
        """
        docstring for continueFrac
        """
        x = self.xyz[0]
        y = x
        array = [1, 1, 3, 5, 3, 1, 1, 10]
        num = len(array)
        for i in range(num):
            x = 1/(array[num - 1 - i]+x)
            x = sympy.simplify(x)
            print(i,x,sympy.latex(x))
        print(x,y)
        s = (273*y + 2885)*y - (155*y + 1638)
        s = sympy.factor(s)
        print(s)
        y = 2885
        x = 5*y + 1638
        return

    def num2Digits(self,num):
        """
        docstring for num2Digits
        num:
            positive integers
        return:
            array for all the digits 
            such as 1234 => [1,2,3,4]
        """
        digits = []
        while num:
            digits.append(num%10)
            num = num // 10
        digits = np.flip(digits)
        return digits

    def getMultiNum(self,num):
        """
        docstring for getMultiNum
        num:
            positive integers
        return:
            a number 
            1234 => [1,2,3,4] => 24 => [2,4] => 8
            1234 => 24 => 8
            the number is 2

        """
        for i in range(100):
            digits = self.num2Digits(num)
            print(i,num,digits)

            num    = np.prod(digits)
            if len(digits) <= 1:
                break
        
        return
    def pythagorean(self):
        """
        docstring for pythagorean
        """
        array = []
        num = 10
        for n in range(1,num):
            for m in range(n+1,num):
                array.append([m+n,m,n])
        array.sort()
        for i,(A,m,n) in enumerate(array):
            a = m**2 - n**2 
            b = 2*m*n 
            c = m**2 + n**2 
            print("%4d%4d%4d"%(min([a,b]),max([a,b]),c))
            
        return

    def inverseNum(self,m):
        """
        docstring for inverseNum
        """
        n = 0
        while m != 0:
            n = 10*n + (m % 10)
            m = m // 10 
        return n

    def testInverseNum(self):
        """
        docstring for testInverseNum
        4*ABCDE = EDCBA
        """
        # for i in range(25,25000):
        #     if self.inverseNum(i) == *i:
        #         print(i)

        factor = 999
        num1   = 1
        num2   = 10001
        for i in range(num1,num2):
            num = self.inverseNum(i*factor)
            j   = num // factor
            # j   = self.inverseNum(j)
            # if i%10 == 0:   
            #     print(i,j,i // 10 + j)
            # else:
            #     print(i,j,i+j)
            k = i // j 
            if i == (k*j) and i != j :
            # if i == (k*j) and i != j and i*factor%10 != 0:
                print(i,j,k,i*factor,j*factor)
        return
    def testAverageProblem(self):
        """
        docstring for testAverageProblem
        (45,a,b,c,d,100)
        The next one is the average of the previous two,
        how much is a?
        """
        x = self.xyz[0]
        y = self.xyz[1]
        a = sympy.solve(x-(x-45)/16-100)
        print("a",a)
        # a = self.one*311/3 
        a = x
        b = 45
        for i in range(4):
            c = a 
            a = (a+b)/2
            a = sympy.expand(a)
            b = c 
            print(i+2,a)
        a = sympy.solve(a-100)
        print("a",a)
        a = sympy.solve(x**2-x/2-1/2)
        print("a",a)
        a = sympy.Symbol("a")
        b = sympy.Symbol("b")
        print(x,y)
        # s = sympy.solve([x**2-y+3,2*x+y**2-10],[x,y])
        # print(sympy.latex(s[0]))
        p = (sympy.sqrt(6177)/9+9107*self.one/27)**(self.one/3)
        q = sympy.sqrt(40*self.one/3+872/p/9+2*p)
        s = sympy.sqrt(-q**2 - 8/q + 40)
        y0 = -(q + s)/2 
        x0 = 5-(y0**2)/2
        print("p = ",p)
        print("q = ",q)
        print("s = ",s)
        k = x0**2 - y0 
        k = sympy.expand(k)
        k = sympy.simplify(k)
        print(k)

        p = (np.sqrt(6177)/9+9107/27)**(1/3)
        q = sympy.sqrt(40/3+872/p/9+2*p)
        s = sympy.sqrt(-q**2 - 8/q + 40)
        y0 = -(q + s)/2 
        x0 = 5-(y0**2)/2
        x0 = sympy.expand(x0)
        print("p = ",p)
        print("q = ",q)
        print("s = ",s)
        print("x0 = ",x0)
        print("y0 = ",y0)
        print(abs(x0))
        
        return

    def isPrime(self,m):
        """
        docstring for isP
        """
        n = int(m**0.5)
        count = 0
        factors = []
        for i in range(2,n+1):
            if m % i == 0:
                count += 1

        if count == 0:
            # print(m,"is prime")
            return True
        else:
            return False
            

    def getFactors(self,m):
        """
        docstring for getFactors
        m:
            positive integers
        return:
            10 => [2,5]
        """
        # n = int(m**0.5)
        count = 0
        factors = []
        for i in range(2,m+1):
            if m % i == 0:
                count += 1
                factors.append(i)
                factors = factors + self.getFactors(m // i)
                break
        return factors

    def testPrime(self):
        """
        docstring for testPrime
        pick any x,y,z (positive integers), 
        how much is the probability for xyz is divided by 100?
        100,1,1 *3
        1,4,25 *6
        10,10,1 *3
        2,5,10 *6
        4,5,5 *3
        20,1,5 *6
        25,2,2 *3
        50,1,2 *6
        no,no,no!!!!
        no less than two 2 and no less than two 5
        p(more than two 2) = 1 - p(no 2) - p(one 2)
         = 1-1/8-3/16 = 11/16
         one 2: (2m+1)(2n+1)(4k+2)
        p(more than two 5) = 1 - p(no 5) - p(one 5)
         = 1 - 
        """
        # a = [[2,0,0],
        #      [0,2,0],
        #      [0,0,2],]
        # b = [[0,1,1],
        #      [1,0,1],
        #      [1,1,0],]
        # a = np.array(a)
        # b = np.array(b)
        # for i in range(3):
        #     for j in range(3):
        #         x = 2**b[i]
        #         y = 5**a[j]
        #         print(x*y)
        num = 1000000
        a = np.random.random((num,3))*10000
        a = a.astype(int)
        a = np.prod(a,axis=1)
        print(sum(a%1000 == 0))
        
        return

    def getMod(self,a,p):
        """
        docstring for getMod
        check if a^{p-1} = 1 (mod p)
        """
        binary = bin(p - 1)[2:]
        length = len(binary)
        if binary[-1] == "1":
            result = a 
        else:
            result = 1
        x = a
        for i in range(length - 1):
            x = x * x
            x = x % p
            if binary[-2-i] == "1":
                result = result*x
                result = result%p

        # print(a,p,result)
        return result
    def getModN(self,a,n,p):
        """
        docstring for getMod
        return a^{n} (mod p)
        """
        binary = bin(n)[2:]
        length = len(binary)
        if binary[-1] == "1":
            result = a 
        else:
            result = 1
        x = a
        for i in range(length - 1):
            x = x * x
            x = x % p
            if binary[-2-i] == "1":
                result = result*x
                result = result%p

        # print(a,p,result)
        return result
    def fermatPrimeTest(self,p):
        """
        docstring for fermatPrimeTest
        p:
            a positive integer
        """
        result = 0
        for i in range(2,1002):
            x = self.getMod(i,p)
            result += x
            if x != 1:
                return False
        if result == 1000:
            print("%d may be prime"%(p))
            return True
        else:
        #     print("%d is not prime!!!!!!!"%(p))
            return False
    def testFermat(self):
        """
        docstring for testFermat
        """
        p = 1<<968
        p = p - 1
        # for i in range(0,10000,2):
        #     s = p + i
        #     print(i)
        #     self.fermatPrimeTest(p+i)

        p = (1<<16) + 1
        a = p
        for i in range(3,1002,2):
            x = self.getMod(i,p)
            if x < a:
                a = x 
            print(i,x)
        print("minimum: ",a)
        print(self.getFactors(p))
        p = 6700417 - 1
        print(self.getFactors(17449))
                
        return
        
    def testBefore(self):
        """
        docstring for testBefore
        """
        print(self.getBinomial(3))
        print(self.getBinomial(4))
        print(self.getBinomial(6))
        divisor = self.getDivisorSeq(39,37)
        print(divisor)

        self.diophantine(1027,712)
        self.remainderTheorem()  
        self.pellSol()
        self.getInitPell(999)
        self.pellSol(31)
        num = 277777788888899
        digits = self.num2Digits(num)
        print(digits)
        self.getMultiNum(num)

        self.remainderTheorem()
        self.pythagorean()

        self.testInverseNum()
        self.testPrime()
        self.isPrime(39252)
        print(self.getFactors(39252))
        self.testAverageProblem()

        self.diophantine() 
        self.bernoulliGen() 
        self.dealData() 
        self.continueFrac() 


        # self.testFermat()
        # self.diophantine(98765432123456789,12345678987654321)
        # print(self.getFactors(98765432123456789))
        # print(self.getFactors(12345678987654321))
        # self.isPrime
        # print(self.fermatPrimeTest(987654321234567834349))
        # self.testCubic()
        # self.hardyWeinberg()
        # self.testAllMod()
        # self.selectNum70()
        # self.polygon17()
        # self.polygon257()

        # self.idCardCheck()
        # self.fermatAndGroup(56)

        # self.primeSpiral()
        # print(self.getModN(2,18000,349589))
        # self.RSA()
        # self.alternatedGroup()
        # self.fibonacci()
        # t1   = time.time()
        # num  = 95
        # s    = self.divideNumber2(num)
        # print(time.time() - t1)
        # self.polyhedron33335()
        # self.polyhedron3454()

        # self.polyhedronTrun()
        # self.getAllPolyhedra()
        # self.laplacian()
        # self.laplacian4D()
        # self.testLaplacian()
        # self.intersection()
        # self.testMatrices()
        # print("solution",self.getCubicSol([1,-27,225,-625]))
        # self.testSeries()
        # self.tangentSeries()
        # self.testBernoulli6()
        # self.testElliptic()

        # self.testSinNX()
        # self.quinticEqn()
        # self.testGetAllCombinator()
        # self.polyRootsPow()
        # print(self.getGeneralCombinator([1,2,3,4]))
        # self.rootTermsNumber()
        # self.getSn()
        # self.getSnByMat()
        # self.getSnByCombinator()
        # self.getSnByComCoef()
        # for n in range(3,21):
        #     self.getSnDirect(n)
        # self.getQuinticTransform()
        # self.getSnByNewton()
        # self.dealQuinticBySn()
        # self.getSnExponent()
        
        return

    def testCubic(self):
        """
        docstring for testCubic
        
        \phi^{3} &=& \phi+1 \\
        \phi    &=& \sqrt[3]{\frac{1}{2}+\frac{1}{6}\sqrt{\frac{23}{3}}}+
                    \sqrt[3]{\frac{1}{2}-\frac{1}{6}\sqrt{\frac{23}{3}}}
        """
        a = (23/3)**0.5/6 
        b = 0.5
        x = (b+a)**(1/3) + (b-a)**(1/3)
        b = 25/54
        y = (b+a)**(1/3) + (b-a)**(1/3)-1/3
        print(a,x,x**3,y,1/x)


        a = ((60*3**0.5+108)/216)**(1/3) + ((-60*3**0.5+108)/216)**(1/3)
        y1 = (60*sqrt(3)+108)/216
        y2 = (-60*sqrt(3)+108)/216
        k  = self.one / 3
        x  = y1**k + y2**k
        x  = simplify(expand(x**3))
        print(a,a**3,2**(1/3))
        print(latex(x))
        x = 3 - sqrt(3)
        x = expand(x**3)
        print(x)


        return
    def hardyWeinberg(self):
        """
        docstring for hardyWeinberg

        initialized by [p,2q,r]
        p_{n+1} = p^{2}
        q_{n+1} = pq
        r_{n+1} = q^{2} 
        p=p_{n}+q_{n},
        q=r_{n}+q_{n}
        """
        p0 = [0.9,0.1,0]
        for i in range(10):
            q1 = p0[0] + p0[1]/2
            q2 = p0[2] + p0[1]/2
            p0 = [q1*q1,2*q1*q2,q2*q2]
            print(i+1,p0)
        
        return
    def getAllMod(self,a,p):
        """
        docstring for getAllMod
        a:
            such as 3,5...
        p:
            such as 5,7...
        return:
            array, (2,5) => (1,2,4,3)
        """
        result = [1]
        for i in range(p-2):
            x = result[-1]*a
            x = x%p 
            result.append(x)
    
        return result
    def getModAdd(self,result,p):
        """
        docstring for getModAdd
        """
        result = self.getOddEven(result)
        num = len(result)
        total = []
        for i in range(num):
            x = result[i,0]+result[:,1]
            x = x%p
            total.append(x)
        total = np.array(total)
        total = total.reshape(-1)
        total.sort()

        numbers = np.zeros(p-1,np.int)
        for i,j in enumerate(total):
            # get the index of z
            k = self.inverseMap[j-1]
            numbers[k] += 1

        # print(result)
        print("numbers",numbers,total)
        return result,total

    def getOddEven(self,result):
        """
        docstring for getOddEven
        """
        result = np.array(result)
        result = result.reshape((-1,2))
        return result
    def testAllMod(self):
        """
        docstring for testAllMod
        """
        a = 3
        p = 65537
        result = self.getAllMod(a,p)
        # (0,1),(1,3) ==> (1-1,0),(3-1,1)
        # inverse map of result
        self.inverseMap = np.zeros(p-1,np.int)
        for i,k in enumerate(result):
            self.inverseMap[k-1] = i

        print(result)
        res,total = self.getModAdd(result,p)
        print("1",res[:,0])
        res,total = self.getModAdd(res[:,0],p)
        print("2",res[:,0])
        res,total = self.getModAdd(res[:,0],p)
        print("3",res[:,0])
        
        
        return
    def polygon17(self):
        """
        docstring for polygon17
        """
        theta = 2*np.pi/17 
        x     = np.arange(1,17)*theta
        x     = np.cos(x)
        print(x,sum(x))

        a0 = -1
        prod =  4*a0
        a1 = (a0 + np.sqrt(a0**2 - 4*prod))/2
        prod =  4*a0
        a2 = (a0 - np.sqrt(a0**2 - 4*prod))/2
        prod =  1*a0
        a5 = (a1 - np.sqrt(a1**2 - 4*prod))/2
        prod =  1*a0
        a6 = (a2 - np.sqrt(a2**2 - 4*prod))/2
        prod =  1*a6
        a13 = (a5 + np.sqrt(a5**2 - 4*prod))/2
        x2 = a13/2

        print("x2:",x2)
        x2 = np.arccos(x2)*17/np.pi
        print("x2:",x2)

        a0 = -self.one
        prod =  4*a0
        a1 = (a0 + sqrt(a0**2 - 4*prod))/2
        prod =  4*a0
        a2 = (a0 - sqrt(a0**2 - 4*prod))/2
        prod =  1*a0
        a5 = (a1 - sqrt(a1**2 - 4*prod))/2
        prod =  1*a0
        a6 = (a2 - sqrt(a2**2 - 4*prod))/2
        prod =  1*a6
        a13 = (a5 + sqrt(a5**2 - 4*prod))/2
        x2 = a13/2
        x2 = simplify(expand(x2))
        print(latex(x2))
        

        return

    def isRepeated(self,array):
        """
        docstring for isRepeated
        array has been sorted
        return:
            bool value, [1,1,2] ==> True
        """
        
        for i in range(len(array) - 1):
            if array[i] == array[i+1]:
                return True 

        return False

    def isSame(self,array):
        """
        docstring for isSame
        return:
            bool, [1,1] => True
        """
        array = np.array(array)
        if sum(array == array[0]) == len(array):
            return True
        return False

    def selectNum70(self):
        """
        docstring for selectNum70

        select 25 out of 70 (1 to 70)
        which should satisfy
        1. any of them is different
        2. all of them are not primes
        3. There are more than two different 
           prime divisors for any of them
        4. the multiplication for each five numbers 
           are the same.
        """

        count = 0
        divisors = []

        results = []
        for i in range(2,71):
            factors = self.getFactors(i)
            # if len(factors) > 1 and (not self.isRepeated(factors)):
            p1 = (not self.isSame(factors)) 
            p2 = (max(factors) <= 11)
            p3 = (i not in [70,35,63])
            if len(factors) > 1 and p1 and p2 and p3:
            # if len(factors) > 1:
                count += 1
                print(count,i,factors)
                if   11 not in factors:
                    results.append(i)
                divisors = divisors + factors 
        print(divisors)
        stati = {}
        for i in divisors:
            if str(i) in stati:
                stati[str(i)] += 1 
            else:
                stati[str(i)]  = 1
        print(stati)

        print(len(results))
        for i1 in range(20):
            for i2 in range(i1+1,20):
                for i3 in range(i2+1,20):
                    for i4 in range(i3+1,20):
                        s = results[i1]*results[i2]*results[i3]*results[i4]
                        k1 = 19958400 // s 
                        k2 = 19958400 %  s 
                        if k1%11 == 0 and k2 == 0 and k1//11 < 7:
                            print(results[i1],results[i2],results[i3],results[i4],k1,k2,s)
        # print(self.getFactors(196883))
        return

    def polygon257(self):
        """
        docstring for polygon257
        """
        
        a0 = -1
        prod = 64*a0
        a1 = (a0 + np.sqrt(a0**2 - 4*prod))/2
        prod = 64*a0
        a2 = (a0 - np.sqrt(a0**2 - 4*prod))/2
        prod = 16*a0
        a3 = (a1 + np.sqrt(a1**2 - 4*prod))/2
        prod = 16*a0
        a4 = (a2 + np.sqrt(a2**2 - 4*prod))/2
        prod = 16*a0
        a5 = (a1 - np.sqrt(a1**2 - 4*prod))/2
        prod = 16*a0
        a6 = (a2 - np.sqrt(a2**2 - 4*prod))/2
        prod = 5*a0 - 1*a1 - 2*a3
        a7 = (a3 + np.sqrt(a3**2 - 4*prod))/2
        prod = 5*a0 - 1*a2 - 2*a4
        a8 = (a4 - np.sqrt(a4**2 - 4*prod))/2
        prod = 5*a0 - 1*a1 - 2*a5
        a9 = (a5 + np.sqrt(a5**2 - 4*prod))/2
        prod = 5*a0 - 1*a2 - 2*a6
        a10 = (a6 - np.sqrt(a6**2 - 4*prod))/2
        prod = 5*a0 - 1*a1 - 2*a3
        a11 = (a3 - np.sqrt(a3**2 - 4*prod))/2
        prod = 5*a0 - 1*a2 - 2*a4
        a12 = (a4 + np.sqrt(a4**2 - 4*prod))/2
        prod = 5*a0 - 1*a1 - 2*a5
        a13 = (a5 - np.sqrt(a5**2 - 4*prod))/2
        prod = 5*a0 - 1*a2 - 2*a6
        a14 = (a6 + np.sqrt(a6**2 - 4*prod))/2
        prod = 1*a1 + 2*a4 + 1*a7 - 2*a8 + 1*a9
        a15 = (a7 + np.sqrt(a7**2 - 4*prod))/2
        prod = 1*a2 + 2*a5 + 1*a8 - 2*a9 + 1*a10
        a16 = (a8 + np.sqrt(a8**2 - 4*prod))/2
        prod = 1*a1 + 2*a6 + 1*a9 - 2*a10 + 1*a11
        a17 = (a9 + np.sqrt(a9**2 - 4*prod))/2
        prod = 1*a2 + 2*a3 + 1*a10 - 2*a11 + 1*a12
        a18 = (a10 + np.sqrt(a10**2 - 4*prod))/2
        prod = 1*a1 + 2*a4 + 1*a11 - 2*a12 + 1*a13
        a19 = (a11 + np.sqrt(a11**2 - 4*prod))/2
        prod = 1*a2 + 2*a5 + 1*a12 - 2*a13 + 1*a14
        a20 = (a12 + np.sqrt(a12**2 - 4*prod))/2
        prod = 1*a1 + 2*a6 + 1*a13 - 2*a14 + 1*a7
        a21 = (a13 - np.sqrt(a13**2 - 4*prod))/2
        prod = 1*a2 + 2*a3 + 1*a14 - 2*a7 + 1*a8
        a22 = (a14 + np.sqrt(a14**2 - 4*prod))/2
        prod = 1*a1 + 2*a4 + 1*a7 - 2*a8 + 1*a9
        a23 = (a7 - np.sqrt(a7**2 - 4*prod))/2
        prod = 1*a2 + 2*a5 + 1*a8 - 2*a9 + 1*a10
        a24 = (a8 - np.sqrt(a8**2 - 4*prod))/2
        prod = 1*a1 + 2*a6 + 1*a9 - 2*a10 + 1*a11
        a25 = (a9 - np.sqrt(a9**2 - 4*prod))/2
        prod = 1*a2 + 2*a3 + 1*a10 - 2*a11 + 1*a12
        a26 = (a10 - np.sqrt(a10**2 - 4*prod))/2
        prod = 1*a1 + 2*a4 + 1*a11 - 2*a12 + 1*a13
        a27 = (a11 - np.sqrt(a11**2 - 4*prod))/2
        prod = 1*a2 + 2*a5 + 1*a12 - 2*a13 + 1*a14
        a28 = (a12 - np.sqrt(a12**2 - 4*prod))/2
        prod = 1*a1 + 2*a6 + 1*a13 - 2*a14 + 1*a7
        a29 = (a13 + np.sqrt(a13**2 - 4*prod))/2
        prod = 1*a2 + 2*a3 + 1*a14 - 2*a7 + 1*a8
        a30 = (a14 - np.sqrt(a14**2 - 4*prod))/2
        prod = 1*a23 + 1*a24 + 1*a25 + 1*a28
        a39 = (a23 - np.sqrt(a23**2 - 4*prod))/2
        prod = 1*a24 + 1*a25 + 1*a26 + 1*a29
        a40 = (a24 - np.sqrt(a24**2 - 4*prod))/2
        prod = 1*a30 + 1*a15 + 1*a16 + 1*a19
        a46 = (a30 + np.sqrt(a30**2 - 4*prod))/2
        prod = 1*a15 + 1*a16 + 1*a17 + 1*a20
        a47 = (a15 - np.sqrt(a15**2 - 4*prod))/2
        prod = 1*a16 + 1*a17 + 1*a18 + 1*a21
        a48 = (a16 - np.sqrt(a16**2 - 4*prod))/2
        prod = 1*a22 + 1*a23 + 1*a24 + 1*a27
        a54 = (a22 + np.sqrt(a22**2 - 4*prod))/2
        prod = 1*a23 + 1*a24 + 1*a25 + 1*a28
        a55 = (a23 + np.sqrt(a23**2 - 4*prod))/2
        prod = 1*a30 + 1*a40 - 1*a46
        a71 = (a39 - np.sqrt(a39**2 - 4*prod))/2
        prod = 1*a22 + 1*a48 - 1*a54
        a111 = (a47 + np.sqrt(a47**2 - 4*prod))/2
        prod = 1*a7 - 1*a15 - 1*a55 - 1*a71
        a239 = (a111 - np.sqrt(a111**2 - 4*prod))/2
        x32 = a239/2
        print(x32)
        print(np.arccos(x32)*257/np.pi)


        return  

    def idCardCheck(self):
        """
        docstring for idCardCheck
        mod 12-2 method
        """
        idCard = 350121191206231210
        weight = [2]
        for i in range(16):
            x = weight[-1]*2 
            x = x%11 
            weight.append(x)
        weight = np.flip(weight)
        print(weight)

        # get digits
        digits = self.num2Digits(idCard)
        print(digits)

        checkNum = 12 - sum(weight*digits[:-1])%11
        checkNum = checkNum%11 
        print(idCard,checkNum)
        if checkNum == digits[-1]:
            print(idCard,"check succeed")
        else:
            print(idCard,"check failed!")
            

        return 
    def fermatAndGroup(self,n):
        """
        docstring for fermatAndGroup
        n and phi(n)
        Fermat's theorem and Euler's theorem
        """
        result = self.getFactors(n)
        print(result)
        # get rid of repeated numbers
        factors = []
        for i in result:
            if i not in factors:
                factors.append(i)
        print(factors)
        elements = np.arange(1,n+1).tolist()
        for i in factors:
            result = []
            for j in elements:
                if j%i != 0:
                    result.append(j)
            elements = result
        print(elements,len(elements))
        # print(A)
        # find the exponent of an element
        species = {}
        inverse = 1
        for i in elements:
            s = i
            count = 1
            while s != 1:
                inverse = s
                s = (s*i)%n 
                count += 1
            print(i,inverse,count)
            key = str(count)
            if key in species:
                species[key].append(i)
            else:
                species[key] = [i]
        print(species)
        res = [1] + species["3"]
        self.getCharacterTable(elements,n)

        # subgroups
        # 13 39  9  5 31  1 27 53 23 19 45 15 41 11 37 33  3 29 55 25 51 47 17 43
        # 1  3   5  9 11 13 15 17 19 23 25 27 29 31 33 37 39 41 43 45 47 51 53 55
        # 15 45 19 23 53 27  1 31  5  9 39 13 43 17 47 51 25 55 29  3 33 37 11 41
        # [1,13]*[3,39] = [3,39]
        # [1,13]*[1,3,5,11,15,17,19,25,29,33,43,47] 2**11 subgroups of  C12
        # [1,15]*[1,3,5,11,13,17,25,27,29,33,37,41]
        # 7*2**11 subgroups of C12
        # [1,13,15,27]*[[1, 3, 5, 11, 29, 33]]
        # 3*4**5 subgroups of C6
        # 1 C8
        # all the subgroups 
        # 
        # a = np.array([1,3,9,11,15,17,19,25,29,33,43,47])
        a = np.array([1,13,15,27])
        b = self.getQuotientGroup(elements, a,n)
        self.getMultiGroup([1,13],[1,15],n)
        c = self.getMultiGroup(a,b,n)
        c.sort()
        print(c,len(c))
        c = self.getMultiGroup(a,[1,43],n)
        c.sort()
        print(c,len(c))

        return
    def getMultiGroup(self,a1,a2,n):
        """
        docstring for getMultiGroup
        """
        a1 = np.array(a1)
        a2 = np.array(a2)
        result = []
        for i in a1:
            for j in a2:
                x = (i*j)%n 
                if x not in result:
                    result.append(x)
        print(result)
        return result
    def getQuotientGroup(self,elements,a,n):
        """
        docstring for getQuotient
        elements:
            array
        a: 
            a subgroup of the elements
        n:
            the modular number
        """
        result = elements
        print(result)
        num = len(elements) // len(a)
        for i in range(num):
            elems = (result[i]*a)%n
            print(elems)
            for k in range(1,len(a)):
                x = elems[k]
                if x in result:
                    result.remove(x)
        print(result)
        return result
    def getCharacterTable(self,array,n):
        """
        docstring for getCharacterTable
        """
        array  = np.array(array)
        result = []
        for i in array:
            result.append((i*array)%n)
        result = np.array(result)
        print(result)
        return

    def primeSpiral(self):
        """
        docstring for primeSpiral
        16 15 14 13
        5   4  3 12
        6   1  2 11
        7   8  9 10
        """
        from PIL import Image
        import time
        primes = [2,3,5,7,11,13,17,19,23]
        factors = np.array([3,5,7,11,13,17,19,23])
        n = 500 
        count = 1
        t1 = time.time()
        for i in range(25,n*n+1,2):
            # none number in factors can 
            # divide i
            if sum(i%factors == 0) == 0:
                if self.isPrime(i):
                    primes.append(i)
                    # self.fermatPrimeTest(i)
        # print(time.time() - t1, primes)
        indeces = np.ones(n*n,"int")*255
        for ii in primes:
            indeces[ii - 1] = 0
        # print(indeces[:1000])
        
        # get the spiral array
        # n should be even
        rotation = [[0,1],[1,0],[0,-1],[-1,0]]
        matrix = np.zeros((n,n),"int")
        x0,y0 = 0,0
        threshold = 1
        rotation_index = 0 
        ii,jj = rotation[rotation_index]
        for i in range(n**2,0,-1):
            matrix[x0,y0] = i
            x1 = x0 + ii
            y1 = y0 + jj
            p1 = (x1 >= n or y1 >= n)
            if p1 or ((not p1) and matrix[x1,y1] != 0):
                rotation_index = (rotation_index + 1)%4 
                ii,jj = rotation[rotation_index]
                x1 = x0 + ii
                y1 = y0 + jj

            x0,y0 = x1,y1
                

        # print(matrix)
        image = np.ones((n,n,3),np.uint8)
        for i in range(n):
            for j in range(n):
                index = matrix[i,j] - 1
                image[i,j,:] = [255,indeces[index],255]

        # print(image[:,:,0])
        image = Image.fromarray(image)
        # image.show()
        image.save("new.png")
        
        return
    def isArrayEven(self,array):
        """
        judge if a array is even permutation
        by computing inverse order
        [0,1,2,3,4]  even
        [0,1,2,4,3]  odd
        [0,1,4,2,3]  even
        """

        n = len(array)
        count = 0 
        for i in range(n-1):
            for j in range(i+1,n):
                if array[i] > array[j]:
                    count += 1
        if count%2 == 0:
            return True 
        else:
            return False

    def getGroupIndex(self,array):
        """
        docstring for getGroupIndex
        hash with n-scale
        """
        n = len(array)
        index = 0 
        for i in array:
            index = n*index + i
        return index

    def getPermuProd(self,arr1,arr2):
        """
        docstring for getPermuProd
        """
        res = []
        assert len(arr1) == len(arr2)
        for i in arr1:
            res.append(arr2[i])
        return res
    def getPermuIdProd(self,id1,id2):
        """
        docstring for getPer
        """
        arr1 = self.elements[id1]
        arr2 = self.elements[id2]
        arr3 = self.getPermuProd(arr1,arr2)
        index = self.getGroupIndex(arr3)
        id3   = self.dicts[str(index)]
        return id3
    def checkGroupTable(self,table):
        """
        docstring for checkGroupTable
        """
        n = len(table)
        baseline = np.arange(n)
        count = 0
        arr = np.zeros(n,int)
        for i in range(n):
            arr[:] = table[:,i]
            arr.sort()
            if sum(arr == baseline) == n:
                count += 1
            arr[:] = table[i,:]
            arr.sort()
            if sum(arr == baseline) == n:
                count += 1
        # print(count,table[1,1],table[:,1].sort())
        if count == 2*n:
            print("it is a group table!!")
        else:
            print("NOT GROUP !!!")
            
        return  
    def alternatedGroup(self):
        """
        docstring for alternatedGroup
        n > 4, A_n is a simple group
        """
        import itertools
        
        # get A5
        n = 4
        self.elements = []
        arr = np.arange(n)
        arr = arr.tolist()
        count = 0
        self.dicts = {}
        for line in itertools.permutations(arr,n):
            if self.isArrayEven(line):
                self.elements.append(line)
                index = self.getGroupIndex(line)
                self.dicts[str(index)] = count 
                count += 1
        # print(elements,len(elements))
        print(len(self.elements))
        # print(dicts,len(dicts))
        # print(self.getPermuProd(elements[3],elements[4]))
        # print(self.getPermuProd(elements[4],elements[3]))
        # print(self.getPermuIdProd(3,4))
        # print(self.getPermuIdProd(4,3))
        group_table = []
        order = np.math.factorial(n)//2
        for i in range(order):
            for j in range(order):
                k = self.getPermuIdProd(i,j)
                group_table.append(k)
        group_table = np.array(group_table)
        group_table = group_table.reshape((order,-1))
        print(group_table)
        # np.savetxt("table.json",group_table,fmt="%d")
        # get cosets
        arr1 = []
        arr2 = []
        for i in range(3,order):
            arr3 = []
            arr4 = []
            for j in range(3):
                k1 = self.getPermuIdProd(i,j)
                k2 = self.getPermuIdProd(j,i)
                arr3.append(k1)
                arr4.append(k2)
                if k1 not in arr1:
                    arr1.append(k1)
                if k2 not in arr2:
                    arr2.append(k2)
            print(i,arr3,arr4)

        arr1.sort()
        arr2.sort()
        print(arr1)
        print(arr2)

        self.checkGroupTable(group_table)
        return

    def RSA(self):
        """
        docstring for RSA
        public key (n,e1)
        private key (n,e2)
        message m, crypted message c
        r = (p-1)(q-1)
        m^e1 = c mod n
        c^e2 = m mod n
        c^e2 = m^(e1*e2)=m^(kr+1)=m mod n 
        """
        p = 131071
        q = 524287
        n = p*q 
        r = (p-1)*(q-1)
        print(self.getFactors(p-1))
        print(self.getFactors(q-1))
        e1 = 11*23*29 
        e2 = self.diophantine(e1,r)[0]
        e2 = e2%r
        print(e1,e2)
        A = 2
        B = self.getModN(A,e1,n)
        print("1",B)
        B = self.getModN(B,e2,n)
        print("2",B)
        print(e2,e1*e2%r)
        return 

    def fibonacci(self):
         """
         docstring for fibonacci
         Fibonacci 1,1,2,3,... 
            
         Pell 1,2,5,12
            P_{2n+1} = P_{n}^2 + P_{n+1}^2
         """
         a = 1  
         p = 1
         b = p  
         for i in range(40):
             c = a + p*b 
             a = b
             b = c 
             print(i+3,c,a**2+b**2)

             
         return 
    def divideNumber2(self,num):
        """
        docstring for divideNumber
        """
        if num%2 == 0:
            return
        elif num == 3:
            return [3]
        else:
            a = num // 2
            res = [] 
            if a%2 == 1:
                res.append(a+1)
                res = res + self.divideNumber(a)
            else:
                res.append(a)
                res = res + self.divideNumber(a+1)
            return res 

    def divideNumber(self,num):
        """
        docstring for divideNumber
        """
        if num%2 == 0:
            return
        else:
            res = [] 
            while num > 3:
                
                a = num // 2
                if a%2 == 1:
                    res.append(a+1)
                    num = a
                else:
                    res.append(a)
                    num = a+1
            res.append(3)
            return res 
        return
    
    def polyhedron(self):
        """
        docstring for polyhedron   
        """
        x = (sqrt(5) - 1)/2 
        x = (2 - x**2)/2
        x = expand(x)
        print(x)
        print((5**0.5+1)/4,np.cos(np.pi/5))
        y = expand(1-x**2)
        y = sqrt(y)
        tan = (10 - 2*sqrt(5))*(6-2*sqrt(5))/16
        tan = expand(tan)
        print(y,tan)
        print("tan pi/5:",np.tan(np.pi/5),np.sqrt(5-2*5**0.5))

        x = sqrt(5)
        y = (x-1)/(3-x)
        print(simplify(y))

        # dedocahedron
        # r: radii of the inscribed sphere
        y = simplify((25+10*x)/100*((x+1)/2)**2)
        print(y)
        # volume
        y = simplify((25+10*x)/10*((x+1)/2))
        print(y)

        return

    def polyhedron3454(self):
        """
        docstring for polyhedron3454
        """
        x = sqrt(5)
        v3 = (15+10*x)*self.one/2
        v4 = 30+15*x
        v5 = 9*(5+2*x)*self.one/4
        volume = (v3 + v4 + v5)*4

        print(volume)

        print((195+98*5**0.5)/12)

        return

    def polyhedron33335(self):
        """
        docstring for polyhedron33335
        """
        k = -(sqrt(5)+1)/16
        sq5 = 5**0.5
        x = self.getCubicSol([1,0,-self.one/2,k])
        # print(x)
        x = self.getValue(54+54*sq5,6*(102+162*sq5)**0.5)/12 
        k1 = x
        print(x,(x**3-x/2)*16-1-5**0.5)

        # new equation
        t = self.xyz[0]
        y = (sqrt(5)+1)/4*(1+1/t)
        y = expand(y**3-y/2-(sqrt(5)+1)/16)
        y = -y*(16*t**3)*(sqrt(5)+1)/2
        y = simplify(y)
        print(y)


        para = [2,-15 - 7*sqrt(5),-21 - 9*sqrt(5),-7 - 3*sqrt(5)]
        x = self.getCubicSol(para)

        # test the solution

        y  = (15+7*sq5+self.getValue(20448+9140*sq5,12*(7137+3192*sq5)**0.5))/6
        k2 = y
        print(k1,k2,1/((sq5-1)*k1-1))
        k3  = k1*(1+k2)**(0.5)
        s1  = 15*(5+2*sq5)
        s2  = - 15*(11+5*sq5)/2
        volume = (20*k3+(s1*k3**2+s2)**0.5)/3
        print(volume)
        print((125+43*sq5)/4)

        # print(self.getFactors(26419))


        # para = [1,-10 - 4*sqrt(5),-18 - 6*sqrt(5),-4 - 4*sqrt(5)]
        # x = self.getCubicSol(para)

        # # test the solution

        # y  = (10+4*sq5) 
        # k2 = y

        return
    def polyhedron33334(self):
        """
        docstring for polyhedron33334
        """
        # 3,3,3,3,4
        x = self.getValue(108*2**0.5,12*66**0.5)/12 
        k1 = x
        print(x,8*x**3-4*x)
        y = 2**0.5*(self.getValue(19,3*33**0.5) - 2)/3 
        k2 = y
        print(y,1/x)
        # \sin a/2
        y = ((8 - self.getValue(19,3*33**0.5))/12)**0.5 
        print(y,x**2+y**2)

        t = self.xyz[0]
        y = sqrt(2)*(1+t)/4/t
        y = simplify(y**3-y/2-sqrt(2)/8)
        print(y)
        q = (2+9*21-27*49)//2 
        p = -9*7 - 1
        det = q**2+p**3
        print(q,p,det,42*42*33)
        print(self.getFactors(det))
        print(self.getFactors(-q))

        y = (self.getValue(566,42*33**0.5) - 1)/21 
        k3 = y
        print(7*y**3+y**2-3*y-1)
        print(y,x,1/(2**1.5*x-1))
        
        t = self.xyz[0]
        y = sqrt(2)*(1+t)/2/t
        z = 4*sqrt(2)*y**3/(sqrt(2)*y-1)
        z = simplify(z)
        y = simplify(y**3-y/2-sqrt(2)/8)
        print(z)

        print(y)

        sol = self.getCubicSol([1,-4,-6,-2])
        sol = simplify(expand(sol**2))
        k4 = (self.getValue(199,3*33**0.5) + 4)/3
        print(sol)
        print(self.getFactors(39898))

        print(k1,k2,k1*k2)
        print(k3,(1/(1-k1**2)-2)/2)
        print(k4,1/(k1*2**0.5-1))
        print(2*k1*(1+k4)**0.5)

        print("k1,k2,k3 and volume")
        k2 = k4 
        k3 = k1*(1+k2)**0.5 
        volume = (8*k3+(12*k3**2-6)**0.5)/3
        print(k1,k2,k3,volume)
        return
    def getValue(self,a,b):
        """
        docstring for getValue
        """
        res = (a+b)**(1/3) + (a-b)**(1/3)
        return res

    def getCubicSol(self,parameters):
        """
        docstring for getCubicSol
        """
        a,b,c,d = parameters
        q = -(2*b**3-9*a*b*c+27*a**2*d)*self.one/2
        p = 3*a*c - b**2 
        delta = sqrt(q**2+p**3)
        x = (-b+(q+delta)**(self.one/3)+(q-delta)**(self.one/3))/(3*a)
        x = simplify(x)
        print(p,q,latex(x))
        return x

    def polyhedron468(self):
        """
        docstring for polyhedron468
        """
        print(np.cos(np.pi/8),np.sqrt(2+2**0.5)/2)
        # self.getCubicSol([1,0,-1,-1])
        # x = self.getValue(108,12*69**0.5)/6
        # print(x,x**3,x+1)

        x = sqrt(2)
        v4 = (3+x)*6
        v6 = (1+x)*18
        v8 = (5+3*x)*6
        volume = (v4+v6+v8)/3
        print(volume)


        return

    def polyhedron4610(self):
        """
        docstring for polyhedron4610
        """
        x = sqrt(5)
        y = 5*(x-1)**2*(10-2*x)
        y = simplify(y/16)
        print(y)

        print(np.tan(2*np.pi/5),(5+2*5**0.5)**0.5)
        y = (15+3*x)*(x+1)**2
        print(simplify(y))

        v4 = (3+2*x)*15
        v6 = (2+x)*45
        v8 = (10+2*x)*15
        volume = (v4+v6+v8)/3/5
        print(volume)

        return

    def polyhedron3434(self):
        """
        docstring for polyhedron3434
        """
        x = sqrt(5)
        y = (1250+410*x)*(25+10*x)
        y = simplify(y/250)
        print(y)
        return

    def polyhedronTrun(self):
        """
        docstring for polyhedronTrun
        truncated polyhedron
        """
        
        x = sqrt(2)
        x = expand((1+x)**3)
        print(x)

        x = sqrt(5)
        y = 5*(x+3)/2 + 3*(5+2*x)
        y = simplify(y/3)
        print(y)


        return

    def getCoefs(self):
        """
        docstring for getCoefs
        """
        sq2 = 2**0.5
        sq5 = 5**0.5
        sq6 = 6**0.5
        k1 = self.getValue(108*2**0.5,12*66**0.5)/12 
        k2 = 1/(sq2*k1-1)
        k3 = k1*(1+k2)**0.5 
        K1 = ((k3**2+1)/3)**0.5
        K1_prime = (K1**2-1/4)**0.5
        K2 = (8*k3+(12*k3**2-6)**0.5)/3

        k1 = self.getValue(54+54*sq5,6*(102+162*sq5)**0.5)/12 
        k2 = 1/((sq5-1)*k1-1)
        k3 = k1*(1+k2)**0.5 
        K3 = ((k3**2+1)/3)**0.5
        K3_prime = (K1**2-1/4)**0.5
        K4 = (20*k3+(15*(5+2*sq5)*k3**2-15*(11+5*sq5)/2)**0.5)/3

        res = [K1,K2,K3,K4]

        return res 

    def getAllPolyhedra(self):
        """
        docstring for getAllPolyhedra
        """
        length = 1.0
        sq2    = 2**0.5
        sq3    = 3**0.5
        sq5    = 5**0.5
        sq6    = 6**0.5

        keys = ["tetrahedron","hexahedron","octahedron","dedocahedron",
                "isocahedron","6,6,5","10,10,3","3,3,3,3,4","3,3,3,3,5",
                "3,5,4,5","4,6,8","4,6,10","4,4,4,3","8,8,3","6,6,3",
                "6,6,4","3,4,3,4","3,5,3,5"]

        polyhedra = {}
        K1,K2,K3,K4 = self.getCoefs()

        radius  = [sq6/4,
                   sq3/2,
                   sq2/2,
                   sq3*(1+sq5)/4,
                   (10+2*sq5)**0.5/4,
                   (58+18*sq5)**0.5/4,
                   (74+30*sq5)**0.5/4,
                   K1,
                   K3,
                   (11+4*sq5)**0.5/2,
                   (13+6*sq2)**0.5/2,
                   (31+12*sq5)**0.5/2,
                   (5+2*sq2)**0.5/2,
                   (7+4*sq2)**0.5/2,
                   22**0.5/2,
                   10**0.5/2,
                   1,
                   (sq5+1)/2]
        volumes = [sq2/12,
                   1,
                   sq2/3,
                   (15+7*sq5)/4,
                   (15+5*sq5)/12,
                   (125+43*sq5)/4,
                   (515+215*sq5)/12,
                   K2,
                   K4,
                   (195+98*sq5)/12,
                   (11+7*sq2)*2,
                   (19+7*sq5)*2,
                   (12+10*sq2)/3,
                   (21+14*sq2)/3,
                   23*sq2/12,
                   8*sq2,
                   5*sq2/3,
                   (45+17*sq5)/6
                   ]
        volumes = np.array(volumes)*length
        radius  = np.array(radius)*length**3
        rates   = [] 
        for i in range(len(radius)):
            R = radius[i]
            V = volumes[i]
            rate = V/(4*np.pi*R**3/3)
            rates.append(rate)
            # print("%2d,%15s,%7.3f,%7.3f,%7.3f"%(i,keys[i],R,V,rate))
            # print("$%7.3fa$ $%7.3fa^3$ %5.1f%s"%(R,V,rate*100,"%"))
            print("%7.3f %7.3f %5.1f"%(R,V,rate*100))

        # keys = np.array(keys)
        # print(keys[np.argsort(rates)])
        # print(np.sort(rates))
        # print(np.sort(volumes))
        return      

    def laplacian(self):
        """
        docstring for laplacian
        compute the deriations
        """
        theta = Symbol("theta")
        phi   = Symbol("phi")
        r     = Symbol("r")
        formulas = [[sin(theta)*cos(phi),cos(theta)*cos(phi)/r,-sin(phi)/(r*sin(theta))],
                    [sin(theta)*sin(phi),cos(theta)*sin(phi)/r,cos(phi)/(r*sin(theta))],
                    [cos(theta),-sin(theta)/r,0]]


        variables = [r,theta,phi]
        self.getLaplacian(variables,formulas)

        

        return
    def laplacian4D(self):
        """
        docstring for laplacian
        compute the deriations
        """
        theta1 = Symbol("theta1")
        theta2 = Symbol("theta2")
        theta3 = Symbol("theta3")
        r      = Symbol("r")
        variables = [r,theta1,theta2,theta3]
        formulas = [r*sin(theta1)*sin(theta2)*sin(theta3),
                    r*sin(theta1)*sin(theta2)*cos(theta3),
                    r*sin(theta1)*cos(theta2),
                    r*cos(theta1)]
        quotients = [1,r**2,(r*sin(theta1))**2,(r*sin(theta1)*sin(theta2))**2]
        dim = len(variables)
        format_string = self.getFormatString(dim)
        # print(format_string)

        # get inverse differential matrix
        tangent = []
        for i in range(dim):
            line = []
            array = []
            for j in range(dim):
                x   = formulas[i]
                var = variables[j]
                y   = diff(x,var)/quotients[j]
                y   = simplify(y)
                array.append(y)
                y   = self.getLatex(y)
                line.append(y)
                # print(j,y)

            tangent.append(array)
            print(format_string % (tuple(line)))
        
        self.getLaplacian(variables,tangent)

    def laplacianAnyD(self,dim):
        """
        docstring for laplacianAnyD
        """
        # get variables
        variables = [Symbol("r")]
        variables += list(sympy.symbols("theta1:%d"%(dim)))
        # for i in range(1,dim):
        #     variables.append(Symbol("theta%d"%(i)))

        # get expressions
        formulas = [variables[0]]
        for i in range(1,dim):
            x = sin(variables[i])
            formulas.append(formulas[-1]*x)
        for i in range(dim-1):
            x = cos(variables[i+1])
            formulas[i] *= x

        # get quotients
        quotients = [1,variables[0]**2]
        for i in range(1,dim - 1):
            x = (sin(variables[i]))**2
            quotients.append(quotients[-1]*x)
        format_string = self.getFormatString(dim)
        # print(variables)
        # print(formulas)
        # print(quotients)
        # get inverse differential matrix
        tangent = sympy.zeros(dim)
        formulas = Matrix(formulas)
        for i in range(dim):
            var = variables[i]
            x   = diff(formulas,var)
            tangent[i,:] = x.transpose()/quotients[i]
        tangent = tangent.transpose()
        # print(tangent)
        # y = tangent[:,0].transpose()*tangent
        # print(trigsimp(tangent.transpose()*tangent))
        self.getLaplacian(variables,tangent)
        return

    def getFormatString(self,dim):
        """
        docstring for getFormatString
        """
        format_string = ""
        for i in range(dim - 1):
            format_string += "%s & "
        format_string += "%s \\\\"

        return format_string

    def getLaplacian(self,variables,formulas):
        """
        docstring for getLaplacian
        variables:
            [r,theta,phi...]
        formulas:
            differential matrix
        """
        dim = len(variables)
        # print(dim)
        format_string = self.getFormatString(dim)

        total = sympy.zeros(dim)[0,:]

        for k in range(dim):    
            tmp = sympy.zeros(dim)
            for i in range(dim):
                var = variables[i]
                x   = diff(formulas[k,:],var)
                tmp[i,:] = x
            # print(k,self.getLatex(simplify(expand_trig(tmp))))
            total += formulas[k,:]*tmp

        print("------------- final results ------------")
        x = expand_trig(total)
        x = simplify(x)
        print(x)
        x = self.getLatex(x)
        print(x)
           
        return

    def getLatex(self,y):
        """
        docstring for getLatex
        """
        # y = trigsimp(y)
        y = latex(y)
        y = y.replace("{\\left("," ")
        y = y.replace("\\right)}","")
        return y

    def testLaplacian(self):
        """
        docstring for testLaplacian
        """
        print("test begins")
        t1 = time.time()
        for i in range(2,5):
            self.laplacianAnyD(i)
            t2 = time.time()
            print(i,t2 - t1)
            t1 = t2


        data = [[2, 0.415179967880249],
                [3, 1.5404701232910156],
                [4, 4.624369859695435],
                [5,11.405637979507446],
                [6,30.010880947113037],
                [7,86.04127788543701]]
        data = np.array(data)
        # data = data[:,1]
        # print(data,data[1:]/data[:-1]) 
        # plt.plot(data[:,0],np.log(data[:,1]))
        # plt.plot(data[:,0]**4,data[:,1],"o-")
        # plt.show()
        return
    
    def intersection(self):
        """
        docstring for intersection
        """
        x = self.xyz[0]
        y = self.xyz[1]
        n = Symbol("n")
        i = Symbol("i")

        y1 = n*(x/(i)+y/(n+1-i))-1
        y2 = y1.subs(i,i+1)
        z = solve([y1,y2],[x,y])
        x1 = z[x]
        y1 = factor(z[y])
        x2 = x1.subs(i,i+1)
        y2 = y1.subs(i,i+1)
        area = -factor(x1*y2 - x2*y1)/2
        print(area)
        y = sympy.summation(area,(i,0,n-1))
        y = factor(y)
        print(y)

        for i in range(1,10):
            print(i,y.subs(n,i))

        z = sympify("x**2 + y**2 - sin(z)")
        z = expand(z**2)
        print(z)

        return

    def testMatrices(self):
        """
        docstring for testMatrices
        """
        x = self.xyz[0]
        y = self.xyz[1]
        A = Matrix([[sin(x),-cos(y)],
                      [-cos(x),sin(y)]])
        B = Matrix([x**2,y**3])
        print(B.transpose()*B)
        print(A)
        a = A.det()
        a = trigsimp(a)

        print(A.diff(x))
        print(a)

        print(A*B)

        print(A.transpose())
        print(A[:,1])
        print(A*A)
        print(trigsimp(A**2))
        s = []
        s.append(latex(A))
        print(s)

        print(latex(A))


        x = sympy.symbols("theta1:3")
        y = Matrix([[sin(x[0])*cos(x[1])],
                    [sin(x[0])*sin(x[1])],
                    [cos(x[0])*sin(x[1])],
                    [cos(x[0])*sin(x[1])]])
        z1 = diff(y,x[0])
        z2 = diff(y,x[1])
        z  = sympy.zeros(4)
        for i in range(2):
            z[i,:]   = z1.transpose()
            z[i+2,:] = z2.transpose()
        print(z)


        return

    def testSeries(self):
        """
        docstring for testSeries
        """
        x = self.xyz[0]
        y = x/(exp(x) - 1)
        for i in range(20):
            s = limit(y,x,0)
            print(y,s)
            y = y.diff(x)
            
        return
    
    def tangentSeries(self):
        """
        docstring for tangentSeries
        """
        x = self.xyz[0]
        y = tan(x)
        # y = y.series(x,0,20)
        # for i in range(10):
        #     y = y.diff(x)
        #     y = expand(y)
        #     print(self.getLatex(y) + "\\\\")
        # initialized by tan(x)
        coef = [0,1]

        bernoulli_index = []
        for i in range(2000):
            # get new coef
            for j in range(len(coef)):
                coef[j] = coef[j]*j 
            # expand the coef
            coef.append(coef[-1])
            n = len(coef)
            for j in range(-1,-n,-2):
                coef[j] = coef[j-1] + coef[j+1]
                coef[j+1] = 0
            # print(j,coef)
            if n%2 == 1:
                coef[0] = coef[1]
                coef[1] = 0
                s = 2**(n-1)
                bernoulli = Integer(coef[0])*(n-1)/s/(s-1)
                # get the numerator and the denominator
                num,denom = sympy.fraction(bernoulli)
                # print(n-1,denom)
                # print(n-2,coef[0],n,d)
                if denom == 66:
                    # print("B%d"%(n-1),num,denom)
                    print("B%d"%(n-1),(n-1)%12,num%denom,denom)
                    bernoulli_index.append(n-1)

            # self.printInverseOdd(coef)
            # print(coef)

        print(bernoulli_index)
        print(len(bernoulli_index))

        return
    def printInverseOdd(self,array):
        """
        docstring for printInverseOdd
        array:
            input array, [1,2,3,4,5] --> 5,3,1
            [1,2,3,4] -> 4,2
        """
        n = len(array)
        if n%2 == 1:
            n = n+1 
        res = []
        for i in range(-1,-n,-2):
            # print(array,i)
            res.append(array[i])
        # print(res)
        return res[-1]


    def testBernoulli6(self):
        """
        docstring for testBernoulli6
        """
        data = np.loadtxt("bernoulli6.json",delimiter=" ",dtype="int")
        # print(data)
        data = data[:,1]
        print(sum(data==2),len(data))

        ii = data[0]
        counts = []
        count  = 1
        for jj in data[1:]:
            if jj == ii:
                count += 1 
            else:
                counts.append(count)
                count = 1
            ii = jj
        counts = np.array(counts)
        counts = counts.reshape((-1,2))
        print(counts)
        stati = np.zeros(10)
        for i in counts[:,0]:
            stati[i] += 1 
        print(stati)
        return

    def getSinNX(self,n):
        """
        docstring for getSinNX
        expand sin(n*x)
        """
        x = self.xyz[0]
        a = cos(x) + sympy.I*sin(x)
        b = cos(x) - sympy.I*sin(x)
        res = expand((a**n - b**n)/2/sympy.I)
        if n%2 == 0:
            res = expand(res/cos(x))
        res = res.subs(sin(x),x)
        res = res.subs(cos(x)**2,1-x**2)
        res = expand(res)
        if n%2 == 0:
            res = res*cos(x)
        print("\\sin(%dx) & = & %s \\\\"%(n,latex(res)))
        return

    def testSinNX(self):
        """
        docstring for testSinNX
        """
        for n in range(3,8):
            self.getSinNX(n)

        return

    def quinticEqn(self):
        """
        docstring for quinticEqn
        """
        a = sympy.symbols("a0:6")
        print(a)
        x = self.xyz[0]
        s = 1 
        for i in range(5):
            s = s*x + a[i+1]
        s = expand(s).collect(x)
        s = s.subs(x,x-a[1]/5)
        s = expand(s).collect(x)
        s = sympy.Poly(s,x).as_dict()
        coef = 15625
        for key in s:
            coef  = coef//5
            value = s[key]
            value = sympy.cancel(value*coef)
            value = latex(value)
            # print(key,value)
        print(s)

        p,q,y,z = sympy.symbols("p q y z")
        y2 = z - p*y - q 
        y3 = expand(y2*y).subs(y**2,y2)
        y3 = expand(y3).collect(y)
        y5 = expand(y2*y3).subs(y**2,y2)
        y5 = expand(y5).collect(y)
        print(y2)
        print(y3)
        print("y5",latex(y5))
        a,b,c,d = sympy.symbols("a b c d")
        s = expand(y5+a*y3+b*y2+c*y+d).collect(y)
        y = solve(s,y)[0]
        print("y",latex(y))
        f1,f2 = sympy.fraction(y)
        print("f1:",latex(f1.collect(z)))
        print("f2:",latex(f2.collect(z)))
        s = f1**2+p*f1*f2+(q-z)*f2**2
        s = expand(-s).collect(z)
        # s = expand(s*f2**2)
        # print(f1,"f1:f2",f2)
        print("s:",latex(s))
        s = sympy.Poly(s,z).as_dict()
        for key in s:
            value = factor(s[key])
            print(key,latex(value))
        # get p and q
        P = s[(4,)]
        Q = s[(3,)]
        q1 = solve(P,q)[0]
        Q = Q.subs(q,q1)
        p = solve(Q,p)[0]

        print(latex(p))
        print(latex(Q.collect(p*(-25))))

        # get X
        print("-----------------------")
        print("get X")
        z,X = sympy.symbols("z X")
        A,B,C = sympy.symbols("A B C")
        b = sympy.symbols("b0:5")
        z4 = X - (z*(z*(z*b[1]+b[2])+b[3])+b[4])
        z4 = expand(z4)
        z5 = expand(z4*z).subs(z**4,z4)
        z5 = expand(z5).collect(z)
        s  = z5+z*(z*A+B)+C
        z3 = Symbol("z3")
        s = s.collect(z).subs(z**3,z3)
        z3 = solve(s,z3)[0]
        z3 = z3.collect(z)
        z4 = z4.subs(z**3,z3)
        z4 = simplify(expand(z4)).collect(z)
        z5 = -A*z**2-B*z-C
        print("z4",latex(z4))
        print("z5",latex(z5))
        print("s",latex(s))
        print("z3",latex(z3))
        z8 = expand(z3*z5).subs(z**3,z3)
        z8 = z8.subs(z**4,z4)
        z8 = simplify(expand(z8)).collect(z)
        print("z8:",latex(z8))
        # z10 = expand(z8*z**2).subs(z**3,z3)
        # z10 = z10.subs(z**4,z4)
        # z10 = expand(z10).collect(z)
        # print("z10:",latex(z10))
        # s  = expand(z10 + 2*C*z5+C*C - A**2*z4 - B*B*z**2 - 2*A*B*z3)
        # s  = expand(s).collect(z)
        # print("z**2,s: ",latex(s))

        return

    def polyRootsPow(self):
        """
        docstring for polyRootsPow
        polynomial (1,a_1,a_2,...)
        roots x1,x2,...
        s1 = sum(x_i)
        sk = sum(x_i^k)
        """
        # a = sympy.symbols("a a b c d")
        a = sympy.symbols("a0:10")
        # print(a)
        s1 = -a[1]
        s2 = s1**2 - 2*a[2]
        s2 = expand(s2)
        s3 = (3*s1*s2-6*a[3]-s1**3)/2
        s3 = expand(s3)
        s4 = 2*(-s1*a[3]-2*a[4]) + (4*s1*s3+3*s2**2-s1**4)/6
        s4 = expand(s4)
        print("s2: ",latex(s2))
        print("s3: ",latex(s3))
        print("s3: ",latex(s4))
        print(self.getBinomial(5))
        
        # for n in range(2,8):
        #     res = self.getCombinatorEqnRecursive(n,n)
        #     sn  = self.getCombinatorForm(res)
        #     print(n,latex(sn))
        # print(res)
        for i in range(2,5):
            # res = self.getCombinatorEqnRecursive2(i,i)
            res = self.getCombinatorEqnRecursive(i,i)
            # print(i,len(res),res)
            print(i,len(res),res)
        
        n = 6
        res = self.getCombinatorEqnRecursive(n,n)
        print(res)
        res,sn = self.getAnotherCombinator(res)
        print(res)
        return
    def getGeneralCombinator(self,arr):
        """
        docstring for getGeneral
        [n1,n2,...] = (N!)/(prod n_i!)
        """
        total = sum(arr)
        total = np.math.factorial(total)
        for i in arr:
            total = total // np.math.factorial(i)
        return total
    def getAnotherCombinator(self,res):
        """
        docstring for getAnotherCombinator
        res:
            2D array
            [[2,0],[0,1]] => [[1,1],[2]]
            2 ones and 1 two
        """
        output = []
        sn = []
        number = ""
        for line in res:
            arr = []
            for i,item in enumerate(line):
                if item > 0:
                    arr += [i+1]*item
            # item = "%d S_{%s} \\\\"
            item = "S_{%s}"
            ch   = str(arr)
            k = self.getGeneralCombinator(arr)
            item = item%(ch)
            number = "%s&%d"%(number,k)
            # print(item)
            output.append(arr)
            sn.append(item)

        # print(sn)
        number = number[1:]
        print("coefficients: ",number)


        return output,sn 
    def getCombinatorForm(self,res):
        """
        docstring for getCombinatorForm
        res:
            2D array
            [[0,1],[1,0]] => a_2 + a_1
        """
        a = sympy.symbols("a0:30")
        sn = 0
        for line in res:
            A = 1
            for j,item in enumerate(line):
                A   *= a[j+1]**item
            print(latex(A) + "\\\\")
            sn += A

        return sn 
    def getCombinatorEqn(self,n):
        """
        docstring for getCombinatorEqn
        """
        a = sympy.symbols("a0:30")
        sn = 0 
        for i in range(1,n+1):
            sn += i*a[i]
        # print(sn)
        arr = []
        for i in range(1,n):
            arr.append(np.arange(n//i+1))
        combinator = self.getAllCombinator(arr)

        sn = 0

        count = 0 
        for i,line in enumerate(combinator):
            res = 0 
            A = 1
            for j,item in enumerate(line):
                res += (j+1)*item
                A   *= a[j+1]**item
            if res in [0,n]:
                j   = 1 - res // n
                # print(i,line)
                sn += A*a[n]**j
                count += 1 
        # print(sn)
        # print("n,count:",n,count)
        return count,sn
    def testGetAllCombinator(self):
        """
        docstring for testGetAllCombinator
        """
        arr = [[0,1],[0,1],[0,1]]
        # arr = [[0,1],[0,1]]
        # arr = [[0,1]]
        res = self.getAllCombinator(arr)
        print(res)
        return

    def getCombinatorEqnRecursive2(self,n,s):
        """
        docstring for getCombinatorEqnRecursive
        integer solutions for
        a1 + a2+...+ a_n = s
        a1 <= a2 <= ... a_n
        In fact, it is equivalent to a1 + 2a2+...+ na_n = s
        """
        if n == 2:
            res = []
            num = s // 2 + 1 
            for x in range(num):
                y = s - x
                res.append([x,y])
            return res
        else:
            total = []
            for y in range(s+1):
                res = self.getCombinatorEqnRecursive2(n-1,y)
                for line in res:
                    if line[-1] <= s - y:    
                        line.append(s - y)
                        total.append(line)

            return total
            
    def getCombinatorEqnRecursive(self,n,s):
        """
        docstring for getCombinatorEqnRecursive
        integer solutions for
        a1 + 2a2+...+ na_n = s
        """
        if n == 2:
            res = []
            num = s // n + 1 
            for y in range(num):
                x = s - 2*y 
                res.append([x,y])
            return res
        else:
            num = s // n + 1
            total = []
            for y in range(num):
                res = self.getCombinatorEqnRecursive(n-1,s-n*y)
                for line in res:
                    line.append(y)
                    total.append(line)
            return total 
        
        
    def getAllCombinator(self,arr):
        """
        docstring for getAllCombinator
        arr:
            2D-array
            [[0,1],[0,1]] => [[0,0],[0,1],[1,0],[1,1]]
        """
        if len(arr) == 1:
            res = []
            for i in arr[0]:
                res.append([i])
            # print("res:",res)
            return res 
        else:
            res = self.getAllCombinator(arr[:-1])
            total = []
            # print("res: ",res)
            for line in res:
                a = line.copy()
                a.append(0)
                # print(line)
                for item in arr[-1]:
                    b = a.copy()
                    b[-1] = item
                    # print(line,"b:",b)
                    total.append(b)
            # print("length:",len(arr))
            # print("total:",total)

            return total
    def getCombinator(self,n,m):
        """
        docstring for getCombinator
        """
        s = 1 
        for i in range(m):
            s = s*(n-i)/(i+1)
        return s 
    def rootTermsNumber(self):
        """
        docstring for rootTermsNumber
        """
        n = Symbol("n")
        s = self.getCombinator(n,5)
        print(s)

        cn5 = self.getCombinator(n,5)
        cn4 = self.getCombinator(n,4)
        cn3 = self.getCombinator(n,3)
        cn2 = self.getCombinator(n,2)
        s = expand(10*cn5+16*cn4+9*cn3)
        s = factor(s)
        print(s)
        s = s - cn2*cn3 
        s = factor(s)
        print(s)

        x,y,z = self.xyz 
        # s1 = x*cn5+4*y*cn4+3*z*cn3
        # s2 = cn2*cn3
        s1 = 4*x*cn4 + 3*y*cn3 + 2*cn2*z
        s2 = 2*cn2*cn2 
        s1 = expand(s1).collect(n)
        print("s1",latex(s1))
        s1 = sympy.Poly(s1,n).as_dict()
        print(s1)
        s2 = expand(s2).collect(n)
        print("s2",latex(s2))
        s2 = sympy.Poly(s2,n).as_dict()
        print(s2)

        eqn = []
        for i in range(2,5):
            eqn.append(s1[(i,)] - s2[(i,)])
        print(solve(eqn,[x,y,z]))
        print(factor(3*cn3+2*cn2))

        s = 120*cn5 + 60*4*cn4 + 50*3*cn3 + 15*2*cn2 + n 
        s = expand(s)
        print(s)
        s = cn2 + 6*cn4 + 6*cn3
        s = factor(s)
        print(s)
        s = 4*cn4 + 5*cn5
        s = factor(s)
        print(s)
        print(factor(2*cn2+3*cn3))
        print("S4",factor(cn2+6*cn3+6*cn4))
        print("cn3+cn4",factor(cn3+cn4))

        # print(factor())

        return

    def getPolyExpansions(self,arr,n):
        """
        docstring for getPolyExpansions
        arr:
            array, [1,1]
        n:
            integer, say 3
        return:
            [1,1],3 => x1x2+x1x3+x2x3
        """
        x = sympy.symbols("x0:%d"%(n+1))
        sn = 0 

        return sn

    def getSn(self):
        """
        docstring for getSn
        """
        n = 5 
        a = sympy.symbols("a0:%d"%(n+1))
        print(a)
        s11111 = -a[5]
        s1112  = -5*s11111 - a[1]*a[4]
        s122   = -10*s11111 - 3*s1112 -a[2]*a[3]
        s113   = -20*s11111-7*s1112-2*s122 -a[1]**2*a[3]
        s23    = -30*s11111-12*s1112-3*s122 - 2*s113 -a[1]*a[2]**2 
        s14    = -60*s11111-27*s1112-12*s122 - 7*s113 -3*s23 -a[1]**3*a[2] 
        s5     = -120*s11111-60*s1112-30*s122 - 20*s113 -10*s23 -5*s14 - a[1]**5

        a = "\\\\"
        # print(latex(s11111) + a)
        # print(latex(s1112) + a)
        # print(latex(s122) + a)
        # print(latex(s113) + a)
        # print(latex(s23) + a)
        # print(latex(s14) + a)
        # print(latex(s5))

        A = np.array([[  1, 0, 0, 0, 0,0,0],
                      [  5, 1, 0, 0, 0,0,0],
                      [ 10, 3, 1, 0, 0,0,0],
                      [ 20, 7, 2, 1, 0,0,0],
                      [ 30,12, 3, 2, 1,0,0],
                      [ 60,27,12, 7, 3,1,0],
                      [120,60,30,20,10,5,1]])
        print(A)
        B = self.getInverseMatrix(A)
        print(B)

        A = np.array([[ 1, 0,0,0,0],
                      [ 4, 1,0,0,0],
                      [ 6, 2,1,0,0],
                      [12, 5,2,1,0],
                      [24,12,6,4,1]])
        print(A)
        B = self.getInverseMatrix(A)
        print(B)   

        A = np.array([[ 1, 0,0],
                      [ 3, 1,0],
                      [ 6, 3,1]])
        print(A)
        B = self.getInverseMatrix(A)
        print(B)               
        
        

        return 

    def getInverseMatrix(self,A):
        """
        docstring for getInverseMatrix
        get the inverse of matrix A with the form
        [[1,...],
         [a1,1,...],
         [a2,a3,1,...],
         [...]]
        """
        n = len(A)
        B = np.zeros((n,2*n),int)
        B[:n,:n] = A 
        B[:n,n:] = np.identity(n)
        for i in range(n-1):
            for j in range(i+1,n):
                B[j] = B[j] - B[j,i]*B[i]

        B = B[:n,n:]
        # if m%2 == 1:
        #     B = - B
        return B

    def getSnByMat(self,n):
        """
        docstring for getSnByMat
        return the combinator sequence
        """
        # get all the sequences
        a = sympy.symbols("a0:%d"%(n+1))
        seq = self.getCombinatorEqnRecursive(n,n)
        # print(res)
        res,sn = self.getAnotherCombinator(seq)
        # print(res)
        # print(sn)
        # print(self.getGeneralCombinator([4,3]))

        # get the matrix of polynomials 
        # A = XB ==> X = AB^{-1}
        polyA = []
        tmp   = []
        polyB = []
        N     = Symbol("n")
        for ii,line in enumerate(res):
            # print(ii,line)
            coef = {}
            for i in line:
                if i not in coef:
                    coef[i] = 1 
                else:
                    coef[i] += 1 

            arr  = list(coef.values())
            num  = self.getGeneralCombinator(arr)
            # print(arr,num)
            polyA.append([len(line),num])
            tmp.append(line)
            
        while len(tmp) > 0:
            polyB.append(tmp.pop())

        # sort the polyA and sn by the length
        num = len(polyA)
        for i in range(num):
            for j in range(i+1,num):
                if polyA[i][0] < polyA[j][0]:
                    polyA[i],polyA[j] = polyA[j],polyA[i]
                    sn[i],sn[j]       = sn[j],sn[i]
        
        return polyA,polyB,sn

    def getCoefMatrix(self,polynomials,n):
        """
        docstring for getCoefMatrix
        """
        m  = n
        N  = Symbol("n")
        n  = len(polynomials)
        # get a null matrix
        matrix = sympy.zeros(n)
        
        i = 0
        for line in polynomials:
            line = sympy.Poly(line,N).as_dict()
            for key in line:
                j = key[0]
                matrix[i,j-1] = line[key]
            i = i + 1           
        matrix = matrix[:,:m]    
        return matrix
    def getSnByCombinator(self):
        """
        docstring for getSnByCombinator
        """

        b     = symbols("b0:5")
        A,B,C = symbols("A B C")
        print(b)
        z  = symbols("z0:3")
        print(z)
        zi = symbols("zi")
        zj = symbols("zj")
        s1 = z[1]**4 + b[1]*z[1]**3 + b[2]*z[1]**2
        s1 += b[3]*z[1] + b[4]
        s1 = zi**4 + b[1]*zi**3 + b[2]*zi**2
        s1 += b[3]*zi + b[4]
        s2 = s1.subs(zi,zj)
        # s2 = s1.subs(z[1],z[2])
        print(s1)
        print(s2)
        # s = expand(s1*s2)
        s = sum(b)
        s = expand(s**3)
        print(latex(s))
        n  = 100 
        # print(self.getCombinator(100,3))
        s = ""
        for i in range(5):
            for j in range(i,5):
                for k in range(j,5):
                    s += self.exponentTerm(i,j,k) + "+"
        s = s[:-1]
        s = sympy.sympify(s)
        s = s.collect(b[4])
        
        print(latex(s))

        return
    def exponentTerm(self,i,j,k):
        """
        docstring for exponentTerm
        [0,0,0] => S_{444}
        """
        a = symbols("b0:5")
        b = [1] + list(a[1:])
        # print(b)
        s = b[i]*b[j]*b[k]
        arr = [i,j,k]
        num = 0
        for i in arr:
            i = 4 - i
            if i > 0:
                num = 10*num + i 
        if num > 0:
            num = self.inverseNum(num)
            res = "%s*S%d"%(str(s),num)
            if num < 10:
                res = "6*" + res
            elif num < 100:
                res = "3*" + res
        else:
            res = "10*%s"%(str(s))
                
        
        return res 

    def getSnByComCoef(self):
        """
        docstring for getSnByComCoef
        get Sn by the combinator coefficients 
        """
        n = 6
        a = symbols("a0:%d"%(n+1))
        A,B,sn = self.getSnByMat(n)
        print("length:",len(A))

        bn = []
        for line in B:
            s = 1 
            for i in line:
                s *= a[i]
            bn.append(latex(s))

        for b,an in zip(bn,sn):
            print("%s & %s \\\\"%(b,an))

        print(" A",A)
        print(" B",B)
        print("sn",sn)
        print("bn",bn)

        An = []
        for i,line in enumerate(B):
            res = self.getCombinatorMultiArr(line)
            print(i+1,res)
            An.append(res)

        # get the coefficients matrix
        n   = len(A)
        num = n*(n-1)//2 
        num = (num + n // 2) // 2
        X = symbols("x0:%d"%(num+1))
        res = self.getConstantMatrix(n,X)
        # get the equations 
        eqn = []
        all_keys = []
        for i in range(n):
            line = {}
            for j,(key,value) in enumerate(A):
                # print(key,value)
                if key not in line:
                    line[key] = value*res[i][j]
                else:
                    line[key] += value*res[i][j]
            # print(line)
            for key in An[i]:
                expr = line[key] - An[i][key]
                if expr != 0:
                    print(expr)
                    all_keys.append(key)
                    eqn.append(expr)
        # print(eqn)
        # print(len(eqn))
        eqn = np.array(eqn)
        all_keys = np.array(all_keys)
        indeces  = np.argsort(all_keys)
        indeces  = np.flip(indeces)
        eqn = eqn[indeces].tolist()
        all_keys = all_keys[indeces]
        print(indeces)
        print(all_keys)
        print(len(eqn),eqn)
        sol = solve(eqn[:30],X[:20])
        print(sol)

        
        return

    def getConstantMatrix(self,n,arr):
        """
        docstring for getConstantMatrix
        """
        res = []
        for i in range(n):
            line = []
            for j in range(n):
                line.append(0)
            res.append(line)
        for i in range(n):
            res[i][i] = 1 
        num = n // 2 
        count = 0 
        for i in range(n):
            for j in range(i+1,n-i):
                count += 1 
                res[j][i] = arr[count]
                res[-i-1][-j-1] = arr[count]
        # res = Matrix(res)
        # print(latex(res))
        return res
    def sortNumKeyDicts(self,dict):
        """
        docstring for sortNumKeyDicts
        """
        keys = dict.keys()
        keys = list(keys)
        keys.sort()
        item = []
        while len(keys) > 0:
            item.append(keys.pop())
        keys = item
        total = {}

        for key in keys:
            total[key] = dict[key]

        return total

    def addCoefDicts(self,dict1,dict2):
        """
        docstring for addCoefDicts
        """
        total = {}
        for key in dict1:
            if key in dict2:
                total[key] = dict1[key] + dict2[key]
            else:
                total[key] = dict1[key]
        for key in dict2:
            if key not in total:
                total[key] = dict2[key]
                
        total = self.sortNumKeyDicts(total)
        return total

    def getCombinatorMultiArr(self,arr):
        """
        docstring for getCombinatorMultiArr
        """
        
        if len(arr) == 1:
            res = {arr[0]:1}
            return res 
        elif len(arr) == 2:
            k,m = arr
            return self.getCombinatorMulti(k,m)
        else:
            k,m = arr[:2]
            res = self.getCombinatorMulti(k,m)
            for k in arr[2:]:
                res = self.getCombinatorMultiDict(res,k)
            return res 

    def getCombinatorMultiDict(self,dict,k):
        """
        dict:
            dict as {1:2,2:34}
        k: 
            integer 
        """
        total = {}
        for key in dict:
            value = dict[key]
            res   = self.getCombinatorMulti(key,k)
            res   = self.dictMulti(res,value)
            total = self.addCoefDicts(total,res)

        return total
    def dictMulti(self,dict,value):
        """
        docstring for dictMulti
        """
        total = {}
        for key in dict:
            num = dict[key]
            total[key] = value * num 

        return total
    def getCombinatorMulti(self,k,m):
        """
        docstring for getCombinatorMulti
        k,m:
            Integer values, 
            k,m => (n,k)*(n,m)
            (n,m) is a combinator number
            C_{n}^{k}C_{n}^{m}
            &=&\sum_{i=0}^{k}C_{m}^{k-i}C_{m+i}^{i}C_{n}^{m+i}
        """
        if k > m:
            k,m = m,k 

        res = {}
        m   = Integer(m)
        k   = Integer(k)
        for i in range(k+1):
            num = self.getCombinator(m,k-i)
            num *= self.getCombinator(m+i,i)
            res[m+i] = num 
        res = self.sortNumKeyDicts(res)
        return res

    def getSmSnMulti(self,m,n):
        """
        docstring for getSmSnMulti
        """
        arr = [1]*m 
        res = self.getSArrSnMulti(arr,n)
        return res

    def getSArr(self,arr):
        """
        docstring for getSArr
        """
        if len(arr) == 1:
            res = {}
            key = [1]*arr[0]
            key = "S_{%s}"%(str(key))
            res[key] = 1 
        elif len(arr) == 2:
            m,n = arr
            res = self.getSmSnMulti(m,n)
        else:
            m,n   = arr[:2]
            res   = self.getSmSnMulti(m,n)
            for k in arr[2:]:
                total = {}
                for key in res:
                    array = self.key2Array(key)
                    # print(key,array)
                    line  = self.getSArrSnMulti(array,k)
                    line  = self.dictMulti(line,res[key])
                    total = self.addCoefDicts(total,line)
                res = total
                # print(k,total)

        
        return res
    def key2Array(self,key_string):
        """
        docstring for key2Array
        'S_{[1, 2, 3]}' => [1, 2, 3]
        """
        key = key_string.split("_")[1]
        key = key[2:-2]
        key = key.split(", ")
        arr = []
        for i in key:
            arr.append(int(i))

        return arr
    def getSArrSnMulti(self,arr,n):
        """
        docstring for getSSnMulti
        arr:
            array, such as [1,2,2]
        n:
            integer, denotes n ones such as [1,1,1...]
        return:
            2D array
            first row: S sequence
            second row: number
            such as [[[1,1,1,1,1,1],[1,1,4]],[15,2]]
        """
        num = min(len(arr),n)
        res = [[],[]]
        a   = np.arange(len(arr))
        for i in range(num+1):
            if i == 0:
                res[0].append([1]*n+arr)
                k = arr.count(1)
                k = self.getGeneralCombinator([k,n])
                res[1].append(k)
            else:
                indeces = itertools.combinations(a,i)
                # print(i,arr)
                for line in indeces:
                    b  = arr.copy()
                    new_array = []
                    # b after addition
                    append_b = []
                    for j in line:
                        b[j] += 1 
                        append_b.append(b[j])

                    # whether is a new array
                    b.sort()
                    c = [1]*(n-i) + b

                    if c not in res[0]:
                        res[0].append(c)
                        k  = self.getNewCoef(append_b,b)
                        # not arr.count
                        k1 = b.count(1)
                        k1 = self.getGeneralCombinator([k1,n-i])
                        # print(append_b,b,k1,k)
                        k  = k*k1
                        res[1].append(k)  
                        # print(k1,k,n,i)
        # 2d array => dict

        total = {}
        for i in range(len(res[0])):
            key   = res[0][i]
            key   = str(key)
            key   = "S_{%s}"%(key)
            value = res[1][i]
            total[key] = value
        
        return total

    def getNewCoef(self,append_b,b):
        """
        docstring for getNewCoef
        append_b,b:
            1D array
        return:
            integer

        [2,2], [1,2,2,2] => 3
        """
        
        # list to dict
        counts = {}
        for i in append_b:
            if i in counts:
                counts[i] += 1 
            else:
                counts[i]  = 1 
        k = 1 
        for i in counts:
            n = b.count(i)
            m = counts[i]

            k = k*self.getGeneralCombinator([m,n-m])
            # print(n,m,k)

        return k
    def getSnDirect(self,n):
        """
        docstring for getSnDirect
        """
        # n = 6
        a = symbols("a0:%d"%(n+1))
        A,B,sn = self.getSnByMat(n)
        B1 = B
        print("length:",len(A))

        bn = []
        for line in B:
            s = 1 
            for i in line:
                s *= a[i]
            bn.append(latex(s))

        # for b,an in zip(bn,sn):
        #     print("%s & %s \\\\"%(b,an))

        coef_dict = {}
        for i,key in enumerate(sn):
            coef_dict[key] = i
        coef_mat = Matrix.zeros(len(A))
        t1 = time.time()
        for i,line in enumerate(B):
            res = self.getSArr(line)
            # t2 = time.time()
            # print(i,"%fs"%(t2 - t1))
            # t1 = t2
            for key in res:
                index = coef_dict[key]
                coef_mat[i,index] = res[key]
        x  = self.getInverseMatrix(np.array(coef_mat))
        x  = Matrix(x)
        if n % 2 == 1:
            x = -x
        # print(latex(x))
        A,B,C = symbols("A B C")
        a = []
        for i in range(n+5):
            a.append(0)
        a[3] = A
        a[4] = B
        a[5] = C
        An = []
        for line in B1:
            s = 1 
            for i in line:
                s *= a[i]
            An.append(s)
        An = Matrix(An)
        # print(An)
        # print(sn)
        y = x*An
        for i,j in zip(sn,y):
            arr = self.key2Array(i)
            # print(arr)
            if len(arr) < 6 and max(arr) < 5:
                print(i,latex(j))

        return

    def getQuinticTransform(self):
        """
        docstring for getQuinticTransform
        """
        b     = symbols("b0:5")
        A,B,C = symbols("A B C")

        S11 =  0
        S21 =  3*A
        S22 =  2*B
        S31 =  4*B
        S32 =  5*C
        S33 =  3*A**2
        S41 =  5*C
        S42 =  -3*A**2
        S43 =  -20*A*B
        S44 =  506*A*C-52*B**2
        

        T2  = S44+b[1]*S43+b[2]*S42+b[3]*S41
        T2 += b[1]**2*S33+b[1]*b[2]*S32+b[1]*b[3]*S31
        T2 += b[2]**2*S22+b[2]*b[3]*S21+b[3]**2*S11
        T2 += (-10*b[4]**2)
        b4 = (4*B + 3*A*b[1])/5
        T2  = expand(T2.subs(b[4],b4)*5)
        T2 = -T2.collect(b[1])
        print(T2)
        print(latex(T2))


        S111 = -A
        S211 = -4*B
        S311 = -5*C
        S411 = 3*A**2
        S221 = -5*C
        S321 = -3*A**2
        S421 = 7*A*B
        S331 = A*B
        S431 = -20*A*C+2*B**2
        S441 = 246*B*C
        S222 = A**2
        S322 = -4*A*B
        S422 = 8*A*C-2*B**2
        S332 = 127*A*C-12*B**2
        S432 = 3*A**3+313*B*C
        S442 = -13235*A**2*B-351434*C**2
        S333 = -A**3-63*B*C
        S433 = 2229*A**2*B+59146*C**2
        S443 = 572416354790*A**2*C-17062513*A*B**2
        S444 = -546*A**4+7120346240107894810*B**3
        S444 += (- 3574023517160095159*A*B*C)

        T3  = S444+b[1]*S443+b[2]*S442+b[3]*S441
        T3 += b[1]**2*S433+b[1]*b[2]*S432+b[1]*b[3]*S431+b[2]**2*S422
        T3 += b[2]*b[3]*S421+b[3]**2*S411+b[1]**3*S333+b[1]**2*b[2]*S332
        T3 += b[1]**2*b[3]*S331+b[1]*b[2]**2*S322+b[1]*b[2]*b[3]*S321
        T3 += b[1]*b[3]**2*S311+b[2]**3*S222+b[2]**2*b[3]*S221
        T3 += b[2]*b[3]**2*S211+b[3]**3*S111 + 10*b4**3
        T3  = expand(T3*25).collect(b[1])
        print("T3",latex(T3))

        return  

    def getInverseSymbolMatrix(self,A):
        """
        docstring for getInverseMatrix
        get the inverse of matrix A with the form
        [[1,...],
         [a1,1,...],
         [a2,a3,1,...],
         [...]]
        """
        n = A.shape[0]
        # print("n = ",n)
        B = sympy.zeros(2*n)
        B = B[:n,:]
        B[:n,:n] = A 
        B[:n,n:] = sympy.eye(n)
        # print(B)
        for i in range(n-1):
            for j in range(i+1,n):
                for k in range(i+1,2*n):
                    # print(B[j,k],B[j,i])
                    B[j,k] = expand(B[j,k] - B[j,i]*B[i,k])
                B[j,i] = 0
        # print("try to expand")
        # B = B.expand()
        # print(np.array(B))
        B = B[:n,n:]
        # if m%2 == 1:
        #     B = - B
        return B
    def getSnByNewton(self,n):
        """
        docstring for getSnByNewton
        """
        # n = 20
        # a = symbols("a0:%d"%(n+1))
        a = []
        for i in range(n+1):
            a.append(0)
        A,B,C = symbols("A B C")
        a[3] = A
        a[4] = B
        a[5] = C
        # print(a)
        M = []
        for i in range(n):
            M.append((i+1)*a[i+1])
        M = -Matrix(M)

        A = sympy.eye(n)
        for i in range(1,n):
            for j in range(n-i):
                A[i+j,j] = a[i]
       
        A_inv = self.getInverseSymbolMatrix(A)
        # print(A_inv)
        # print("inverse:",latex(A_inv))
        S = A_inv*M
        S = S.expand()
        S = [0] + list(S)
       
       
        
        return S 
    def dealQuinticBySn(self,):
        """
        docstring for dealQuinticBySn
        """
        S = self.getSnByNewton(20)
        b = symbols("b0:5")
        A = [0,b[1],b[2],b[3],b[4]]
        b = A
        # print(S[20])
        z = self.xyz[2]
        X = z**4+b[1]*z**3+b[2]*z**2+b[3]*z+b[4]
        # print(X)
        X2 = X
        n  = 4
        for i in range(2,n+1):    
            X2 = expand(X2*X).collect(z)
        #     print("S%dX"%(i),latex(X2))
        s1 = X2
        s1 = s1.subs(b[4]**i,5*b[4]**i)
        for i in range(4*i,0,-1):
            s1 = s1.subs(z**i,S[i])
        # print(latex(s1))
        A,B,C = symbols("A B C")
        # b4 = (3*A*b[1] + 4*B)/5
        # s1 = s1.subs(b[4],b4)*5
        # b2 = -(4*B*b[1]+5*C)/(3*A)
        # s1 = s1.subs(b[2],b2)*9*A**2
        # s1 = expand(-s1*5).collect(b[3])
        print(latex(s1))

        


        return
    def getSnExponent(self):
        """
        X = z**4+b[1]*z**3+b[2]*z**2+b[3]*z+b[4]
        """
        i = 4 
        res = self.getCombinatorEqnRecursive(4,2)
        # res,sn = self.getAnotherCombinator(res)
        print(res)
        return

    def modularEquation(self):
        """
        docstring for modularEquation
        """
        u,v = symbols("u v")
        print(u,v)

        s =  u**6 - v**6 + 5*u**2*v**2*(u**2 - v**2)
        s += 4*u*v*(1-u**4*v**4) 
        print(s)
        # print(solve(u**5-u+1))

        x,y,z,a,k = symbols("x y z a k")
        s  = x**3 + y**3 - x*y
        s  = s.subs(x,k*(u+v))
        s  = s.subs(y,k*(u-v))
        s  = expand(s)
        print(latex(s))
        return
    
    def weierstrassForm(self):
        """
        docstring for weierstrassForm
        """
        x,y = self.xyz[:2]
        u,v,k = symbols("u v k")
        s = x**3 + y**3 - x*y 
        # s = s.subs(x,u+v)
        # s = s.subs(y,u-v).expand()
        u = u/3/4+Integer(3)/4
        x = (1/u - 1)/6 + v/u/27/8
        y = (1/u - 1)/6 - v/u/27/8
        s = x**3 + y**3 - x*y 
        s = (s*108*432*u**3).simplify()
        x = x.simplify()
        y = y.simplify()
        print(latex(x))
        print(latex(y))
        
        print(s,latex(-s))
        u = Symbol("u")
        s = -u**3 + 27*u - 54
        print(s,factor(s))


        # s = -4*u**3 + 9*u**2 - 6*u + 1
        # s = s.subs(u,u+Integer(3)/4).expand()
        # print(s)

        # s = (2*u**3 - u**2)/(6*u + 1)
        # s = s.subs(u,(u-1)/6)
        # s = (s*108*u).expand()
        # print("s: ",latex(s))

        x,y = self.xyz[:2]
        p = symbols("a b c d e f g h i j")
        print(p)
        exponents = [[3,0],[0,3],[2,1],[1,2],[1,1],
                     [2,0],[0,2],[1,0],[0,1],[0,0]]
        s = 0 
        for i,(ii,jj) in enumerate(exponents):
            s += p[i]*x**ii*y**jj
            # print(i,ii,jj)
        print(s)
        # print(u,v)
        theta = Symbol("theta")
        u0,v0 = symbols("u0 v0")
        x1 = u+u0
        y1 = v+v0
        x1 = u*cos(theta) - v*sin(theta)
        y1 = u*sin(theta) + v*cos(theta)
        # print(x1,y1)
        s = s.subs(x,x1)
        s = s.subs(y,y1).expand().collect([u,v])
        s0 = Poly(s,[u,v]).as_dict()
        print(s)
        s = latex(s)
        s = s.replace("\\left(\\theta \\right)","\\theta")
        # print(latex(s))
        s0 = s0[(3,0)]
        print(s0)
        s0 = s0.subs(cos(theta),1)
        s0 = s0.subs(sin(theta),x).collect(x)
        print(s0)

        # k = 1/Integer(2)
        # A = 1/sqrt(k**2+1)
        # B = k/sqrt(k**2+1)
        A = 2
        B = 1
        x1 = u*A - v*B
        y1 = u*B + v*A
        s  =  x**3 + 2*y**3 - 1
        s  = s.subs(x,x1)
        s  = s.subs(y,y1).expand()
        print(s)
        print(latex(s))
        
        s = x**3 - 27*x+54
        print(factor(s))
        for i in range(4):
            for j in range(2):
                a = (3**(i)*2**(j))
                print(a,factor(s-a**2))
        x = [Integer(0),Integer(54)]

        # for i in range(2):
        #     k = (3*x[0]**2 - 27)/(2*x[1])
        #     a = k**2 - 2*x[0]
        #     x[1] = -(k*(a - x[0]) + x[1])
        #     x[0] = a 
        #     print(x,sqrt(x[0]**3-27*x[0]+54))
        t = Symbol("t")
        x = t/(1+t**3)
        y = t**2/(1+t**3)
        print(x,y)
        s1 = (-3*u+v+9)/(18*u+162)-x
        s2 = (-3*u-v+9)/(18*u+162)-y
        sol = solve([s1,s2],[u,v])
        # print(latex(sol))
        # print(latex(sol[v].factor()))
        u = sol[u].factor()
        v = sol[v].factor()
        print("u,v",u,v)
        print(latex(u),latex(v))
        s = factor(u**3-27*u+54)
        print(s)
        points = []
        for i in range(-100,100):
            if i == -1:
                continue
            i = Integer(2)/(2*i+1)
            u1 = u.subs(t,i)
            v1 = v.subs(t,i)
            if u1.is_integer and v1.is_integer:
                print("-------",i)
                print(u1,v1)
                points.append([u1,v1])
        print(points)
        # for x in points:
        #     print("=============",x)
        #     for i in range(2):
        #         k = (3*x[0]**2 - 27)/(2*x[1])
        #         a = k**2 - 2*x[0]
        #         x[1] = -(k*(a - x[0]) + x[1])
        #         x[0] = a 
        #         print(x,sqrt(x[0]**3-27*x[0]+54))
        m = Symbol("m")
        s = 3*(t**2-10*t+1)-m*(1+t)**2 
        s = s.expand().collect(t)
        print(s)
        delta = (-2*m - 30)**2 - 4*(3 - m)**2 
        delta = delta.factor()
        print(delta)

        s = 3*(t**2-10*t+1)-(m**2-6)*(1+t)**2 
        print(s.factor())
        s = solve(s,t)
        print(latex(s))
        # t = s[1]
        print(v)
        v = v.subs(t,s[1]).factor()
        print(v)
        u = m**2 - 6 
        v = m*(m - 3)*(m + 3)
        print(u,v)
        x = (-3*u+v+9)/(18*u+162)
        y = (-3*u-v+9)/(18*u+162)
        t = -(m+3)/(m-3)
        x = t/(1+t**3)
        y = t**2/(1+t**3)
        x = x.factor()
        y = y.factor()
        print(x)
        print(y)

            
        
        return
    def getThirdXY(self):
        """
        docstring for getThirdXY
        """
        t  = Symbol("t")
        s = s.subs(x,t/(1+t**3))
        s = s.subs(y,t**2/(1+t**3)).simplify()
        x0 = Integer(2)/9
        y0 = Integer(4)/9
        l = (3*x0**2 - y0)*(x-x0) + (3*y0**2 - x0)*(y-y0)
        k = -(3*x0**2 - y0)/(3*y0**2 - x0)
        print(l*81/2)
        s = x**3 + y**3 - x*y
        # s = s.subs(y,(12*x+4)/15).expand()
        x0,y0,k = symbols("x0 y0 k")
        s = s.subs(y,k*(x-x0)+y0).expand().collect(x)
        # print(s)
        # t  = Integer(4)
        t = symbols("t0:3")
        x0 = t[1]/(1+t[1]**3)
        y0 = t[1]**2/(1+t[1]**3)
        x1 = t[2]/(1+t[2]**3)
        y1 = t[2]**2/(1+t[2]**3)
        # k = -(3*x0**2 - y0)/(3*y0**2 - x0)
        k  = (y1-y0)/(x1-x0)
        k = k.simplify()
        # print("k",k,latex(k))
        sx = (3*k**3*x0-3*k**2*y0+k)/(k**3+1)
        sx = sx.simplify()
        # print("sx",latex(sx))
        # print(sx.factor())
        x3 = sx - x0 - x1
        x3 = x3.simplify()
        y3 = -(k*(x3-x0)+y0).simplify()
        print("x3",x3)
        print("y3",y3)
        # print(s)
        return

    def getWeierstrassForm(self):
        """
        docstring for getWeierstrassForm
        """
        u,v = symbols("u v")
        x,y = symbols("x y")
        s1 = 12/(u+v) - x
        s2 = 36*(u-v)/(u+v) - y
        print(solve([s1,s2],[u,v]))
        x = (u + 36)/(6*v)
        y = (36 - u)/(6*v)
        s = x**3 + y**3 - 1 
        s = s.expand().simplify()
        print(s)

        points = np.array([[1,1,2],
                           [2,4,9],
                           [3,9,28]])
        x = symbols("x y z")
        u = symbols("u v w")
        tangents = sympy.zeros(3)
        tangents[:,0] = 3*points[:,0]**2 - points[:,1]*points[:,2]
        tangents[:,1] = 3*points[:,1]**2 - points[:,0]*points[:,2]
        tangents[:,2] = - points[:,0]*points[:,1]

        print(tangents)
        sn = []
        for i in range(3):
            s = 0 
            for j in range(3):
                s += tangents[i,j]*(x[j] - points[i,j])
            sn.append(s - u[i])
        print(sn)
        sol = solve(sn,list(x))
        print(sol)
        u = [0,0,0]
        for i in range(3):
            u[i] = sol[x[i]]
        s = u[0]**3 + u[1]**3 - u[0]*u[1]*u[2]
        s = s.expand().simplify()
        print(s)


        return

    def conicSection(self):
        """
        docstring for conicSection
        ellipse:
            x**2/a**2 + y**2/b**2 = 1 
        hyperbola:
            x**2/a**2 - y**2/b**2 = 1 
        parabola:
            y**2 = 2*p*x
        """
        
        k,m    = symbols("k m")
        x0,y0 = symbols("x0 y0")
        x,y   = symbols("x y")
        a,b,p = symbols("a b p")

        y = k*x + m
        # s = x**2/a**2 + y**2/b**2 - 1
        # s = s*a**2*b**2
        s = y**2 - 2*p*x
        s = s.expand().collect(x)
        print(latex(s))
        print(s)
        # ax = -2*a**2*k*m/(a**2*k**2 + b**2)
        # bx = (a**2*m**2-a**2*b**2)/(a**2*k**2 + b**2) 
        ax = (2*p - 2*k*m)/k**2
        bx = m**2/k**2
        ay = k*ax + 2*m 
        by = k**2*bx + k*m*ax + m**2 
        s  = by - y0*ay + y0**2+x0**2 
        s += bx - x0*ax
        s  = s.simplify().factor().collect([m])
        print("final",latex(s))

        return
    def testABCElliptic(self,n,m):
        """
        docstring for testABCElliptic
        
        """
        # n = 6
        arr = np.arange(1,n)
        combinators = itertools.combinations(arr,2)
        count = 0
        arr = []
        # m = 4
        for line in combinators:
            count += 1
            line   = list(line)
            # line[0] = -line[0]
            arr.append([-line[0],line[0]+m,line[1]])

        print("m = ",m,count)
        arr = np.array(arr)
        # print(arr[:10])
        # print(arr.shape)
        S01 = arr[:,0] + arr[:,1]
        S02 = arr[:,0] + arr[:,2]
        S12 = arr[:,1] + arr[:,2]
        S   = S01*S02*arr[:,0] + S01*S12*arr[:,1] 
        S  += S12*S02*arr[:,2]
        k   = S01*S02*S12
        res = S%k
        indeces = np.nonzero(res == 0)[0]
        # print(indeces,type(indeces),indeces[0])
        for i in indeces:
            s = S[i]//k[i]
            if s > 0:
                print("[%d,%d,%d,%d]"%(*(arr[i]),s))
        return
    def getResABC(self,s):
        """
        docstring for getResABC
        """
        res  = Integer(s[0])/(s[1]+s[2])
        res += Integer(s[1])/(s[0]+s[2])
        res += Integer(s[2])/(s[0]+s[1])
        return res

    def testABC(self):
        """
        docstring for testABC
        [-104  105  181] 182
        [-143  145  323] 162
        [-203  205  288] 146
        [-65  68 179] 60
        [-217  220  263] 92
        [-39  49 116] 12
        [-513  517  623] 160
        """
        
        arr =  [[-104, 105, 181,182],
                [-143, 145, 323,162],
                [-203, 205, 288,146],
                [-65, 68,179,60],
                [-217, 220, 263,92],
                [-39, 49,116,12],
                [-513, 517, 623,160],
                [-1400,1406,3718,620],
                [-2581,2587,3465,580],
                [-533,540,2099,300],
                [-901,905,1321,332],
                [-1411,1421,3476,348],
                [-2327,2331,3755,940],
                [-224,235,359,34],
                [-992,1003,1043,114],
                [-1292,1315,1307,144],
                [-1455,1456,2521,2522],
                [-2583,2585,5779,2890],
                [-6929,6931,9800,4902],
                [-1,4,11,4],
                [-5,7,8,6],
                [-7,9,19,10],
                [-14,16,26,14]
                ]
        for line in arr:
            res = self.getPQElliptic(line)
            print(res)
  
        a,c,m,n = symbols("a c m n")

        s = ((a+m)/(c-a)-a/(a+m+c)-n/m)*(c-a)*(c+a+m)*m
        s = s.expand().simplify().collect(a)
        print(latex(s))

        x = (2*n*c + m*n-m**2)/m**2 
        y = (2*a+m)/m**2
        s = x**2 - n*(2*m+n)*y**2 
        s = (s*m**4*(-Integer(1)/4/n)).expand().simplify().collect(a)
        print(latex(s))
        s = m**3 + (m*n-m**2)**2/4/n - m**2*(2*m+n)/4
        s = s.expand()
        print(s)

        n = 1
        m = 3
        self.getABCByNM(n,m)

        x,y = symbols("x y")
        eqn = [16*x+352*y-3712,5*x+133*y-1403]
        print(solve(eqn,[x,y]))
        # print(self.pellSol(2,20))
        

        sol = diophantine(x**2-n*(2*m+n)*y**2-m**4)
        sol = diophantine(x**2-2*y**2-2)
        # print(sol)
        count = 0
        for key in sol:
            count += 1
            print(latex(key[0]) + "\\\\")
        # print(type(sol))
        # print(sol[0])
        arr = [[1,11],[65,179],[700,1859]]
        for a,c in arr:
            # print(a,c)
            x = 2*c - 6 
            y = 2*a + 3 
            print(x,y,x**2 - 7*y**2)


        a,c,m,n,k = symbols("a c m n k")

        s = ((a+m)/(c-a)-a/(a+m+c)+c/m-k)*(c-a)*(c+a+m)*m
        s = s.expand().simplify().collect(a)
        print(latex(s))

                
        return

    def getGeneralPellSol(self):
        """
        docstring for getGeneralPellSol
        """
        n = 1 
        m = 3
        D = 2*m+n
        x = self.pellSol(D,20)
        # print(x)
        # if x:
        #     print(x[0]**2 - D*x[1]**2)
        xm = self.naivePellSol(D,num=m**4)
        # print(x)
        a = Integer(8)/3
        b = Integer(5)/3
        res = [[a,b],[b,a]]
        # for line in xm:
        #     l = self.getPellLambda(line,x[1])
        #     res.append(l)
        #     print(line,l)
        for line in res:
            sol = self.getNewPellSol(x[:10],line)
            for u,v in sol:
                if u%3 == 1:
                    c = u//2 + 3 
                    a = -(v-3)//2
                    b = -a+3
                    k = (c+1)//3
                    print("%d,%d,%d,%d"%(a,b,c,k))
                    print(self.getResABC([a,b,c]))

        n = 3 
        m = 1
        D = n*(2*m+n)
        print(self.pellSol(D,20))

        return

    def getNewPellSol(self,arr,l):
        """
        docstring for getNewPellSol
        """
        res = []
        length = len(arr)
        for i in range(1,length):
            line = []
            for j in range(2):
                line.append(l[0]*arr[i][j] + l[1]*arr[i-1][j])
            res.append(line)
                
        return res
    def getPellLambda(self,arr1,x1):
        """
        docstring for getPellLambda
        arr1:
            1D array with two elements [u,v]
        x1:
            1D array with two elements [x1,y1]

        arr1 = [l1 + x1*l2,y1*l2]
        """
        l2 = arr1[1]/Integer(x1[1])
        l1 = arr1[0] - l2*x1[0]
        return [l1,l2]
    def naivePellSol(self,D,num=1):
        """
        docstring for naivePellSol
        """
        n = 10000
        res = []
        # for y in range(0,n+1):
        #     x = sqrt(num+D*y*y)
        #     if x.is_integer:
        #         res.append([x,y])
        #         # return [x,y]
        arr = np.arange(n+1)
        X2  = D*arr**2 + num 
        X   = X2**0.5 
        X   = X.astype(int)
        indeces = np.nonzero(X2 == X**2)[0]
        for i in indeces:
            res.append([X[i],arr[i]])
        if len(res) == 0:
            print("There is no integer solutions within %d"%(n))
        
        return res
    def getABCByNM(self,n,m):
        """
        docstring for getABCByNM
        n,m:
            two positive integers
        """
        D = n*(2*m+n)
        print(n,m,sympy.factorint(D))
        sol = self.pellSol(D,20)
        print(sol)
        # print(x)
        for line in sol:
            c = m*m*line[0] - m*n + m**2 
            a = m*m*line[1] - m
            # k = c + n
            # print(c,a)
            if a%2 == 0 and c%(2*n) == 0:
                a = -a // 2 
                c = c // (2*n)
                b = -a + m
                k = c + n 
                if k%m == 0:
                    k = k // m
                    print("%d & %d & %d & %d \\\\"%(a,b,c,k))
                    # print(self.getResABC([a,b,c]))
        return
    def checkElliptic(self):
        """
        docstring for checkElliptic
        """
        x,y,k = symbols("x y k")
        a,b,c = symbols("a b c")
        z = (k+2)*(a+b)-c
        x = -4*(k+3)*(a+b+2*c)/z 
        y = 4*(k+3)*(2*k+5)*(a-b)/z 
        p = 4*k*(k+3) - 3
        q = 32*(k+3)
        s = x**3+p*x*x+q*x - y**2
        s = s.expand().factor()
        print(s)

        x,y,k = symbols("x y k")
        a,b,c = symbols("a b c")
        
        a = x-y-8*(k+3)
        b = x+y-8*(k+3)
        c = (2*k+4)*x+8*(k+3)
        s1 = a+b+c 
        s2 = a**2+b**2+c**2 
        s3 = a**3+b**3+c**3
        s  = k*s3 - (k-1)*s1*s2 + (3-2*k)*a*b*c 
        s  = s/(8*(k + 3)*(2*k + 5))
        s  = s.simplify().collect(x)
        print(s)
        a,b,c = symbols("a b c")
        A = x-y-8*(k+3)
        B = x+y-8*(k+3)
        C = (2*k+4)*x+8*(k+3)
        eqn = [A/C-a/c,B/C-b/c]
        sol = solve(eqn,[x,y])
        x   = sol[x].factor()
        y   = sol[y].factor()
        print(latex(x))
        print(latex(y))
        return
    def getPQElliptic(self,arr):
        """
        docstring for getPQElliptic
        """
        a,b,c,k = arr
        z = Integer((k+2)*(a+b)-c) 
        x = -4*(k+3)*(a+b+2*c)/z 
        y = 4*(k+3)*(2*k+5)*(a-b)/z 
        p = 4*k*(k+3) - 3
        q = 32*(k+3)
        # print(y**2,x**3+p*x*x+q*x)
        return [k,x,y,p,q]

    def generalConic(self):
        """
        docstring for generalConic
        """
        a,c,r,theta = symbols("a c r theta")
        s = 1024*a**4*((r**2+c**2)**2 - 4*c**2*r**2*(cos(theta))**2)
        s -= ((32*a**2*(r**2+c**2) - (4*a**2+2*c**2+r**2)**2)**2)
        s = s.expand().collect(r)
        print(s)
        s = sympy.Poly(s,r).as_dict()
        for i in range(0,10,2):
            print(i,"formulas",latex(s[(i,)].factor()))

        l2 = (4*a**2+2*c**2+r**2)**2
        l1 = 64*a**2*(c**2+r**2)
        s  = (64*a**2*c*r*cos(theta))**2 
        s  -= (l1 - l2)*l2 
        s  = s.subs(a,2)
        s  = s.subs(c,1)
        print(latex(s))

        return

    def conicProp(self):
        """
        docstring for conicProp
        """
        a,b,c,x,y,k,m = symbols("a b c x y k m")
        x = k*y - c
        s = (x/a)**2 + (y/b)**2 - 1 
        s = (s*a**2*b**2).expand().collect(y)
        print(latex(s))
        x1,x2,y1,y2 = symbols("x1 x2 y1 y2")
        x,y = self.xyz[:2]
        s1 = x1*x/a**2 + y1*y/b**2 - 1
        s2 = x2*x/a**2 + y2*y/b**2 - 1
        sol = solve([s1,s2],[x,y])
        print(latex(sol[x]))
        print(latex(sol[y]))

        x = k*y 
        s = (x/a)**2 + (y/b)**2 - 1 
        s = (s*a**2*b**2).expand().collect(y)
        print(latex(s))
        x0,y0 = symbols("x0 y0")
        # y1 = (m-c)*y0/(x0-c-k*y0)
        # y2 = (m+c)*y0/(x0+c-k*y0)
        # s1 = y1*y2 - b**2*(m**2 - a**2)/(a*a+(b*k)**2)
        # s1 = (s1*(x0-c-k*y0)*(x0+c-k*y0)*(a*a+(b*k)**2))
        # s1 = s1.expand().collect([m,k])
        # print(latex(s1))

        y1 = -b**2*y0/(a**2+c**2+2*c*x0)
        y2 = -b**2*y0/(a**2+c**2-2*c*x0)
        k1 = (x0+c)/y0
        k2 = (x0-c)/y0
        x1 = k1*y1 - c
        x2 = k2*y2 + c
        m  = x1 - (x2 - x1)*y0/(y2-y1)
        m  = (m*b**2*x0*(a**2+c**2+2*c*x0)).expand().simplify()
        m  = m.subs(b**2,a**2-c**2).expand()
        m  = m.collect(x0)
        print("+++++++++++++++++++++++++++++++++++++++++++++++")
        m  = m.subs(x0,c).subs(b**2,a**2-c**2).expand()
        print(latex(m.factor()))
        y0 = b**2/a
        x1 = c 
        y1 = -y0 
        y2 = -b**4/a/(a**2+3*c**2)
        x2 = -3*c*(3*a**2+c**2)/(a**2+3*c**2)

        m  = x1 - (x2 - x1)*y0/(y2-y1)
        m  = m.subs(b**2,a**2-c**2)
        m  = m.simplify()
        print("    ")
        print(latex(m))

        x,y = self.xyz[:2]
        print(diophantine(x**2-3*y**2-2))
        # print(self.getInitPell(3,num=2))
        print(solve(a**3+a**2-392))


        return


    def quadraticResidue(self):
        """
        docstring for quadraticResidue
        """
        for i in range(20,30):
            p = sympy.prime(i)
            k = (p - 1) // 2 
            # print(i,"test t")
            t = (2**k)%p
            if t != 1:
                t = t - p
            print(i,p,t)

        print("when p is 17")
        p = 17 
        for i in range(1,p):
            t = (i**8)%p 
            print(i,t)

        print(self.getQradraticRes(5,17))    
        print(self.getQradraticRes(17,5))    
        print(self.getQradraticRes(2,5)) 
        print(self.getFactors(2017))   
        return
    def getQradraticRes(self,a,p):
        """
        docstring for getQradraticRes
        """
        k = (p - 1) // 2 
        res = (a**k)  % p 
        if res == (p-1):
            res = res - p 
        return  res
    def getQradraticResByReci(self,a,p):
        """
        docstring for getQradraticResByReci
        get the quadraticResidue by 
        the quadratic reciprocity law
        """
        divisors = self.getModuloSeq(a,p)
        print(divisors)
        # waiting to be done
        # how to do when even numbers occured?
        
        return

    def huiwenCheck(self,n,iternum = 1000):
        """
        docstring for huiwenCheck
        palindromic number 
        1331,55533555 are palindromic numbers 
        """
        # n = 1961

        for i in range(iternum):
            m = self.inverseNum(n)
            if m == n:
                # print(i,m)
                break
            else:
                # print(i,n)
                n += m
        return i 

    def huiwenTest(self):
        """
        docstring for huiwenTest
        """
        for i in range(11,20):
            k = self.huiwenCheck(i)
            # print(i,k)
            if k > 100:
                print(i,k)
        n = 689
        print("test %d"%(n))
        print(self.huiwenCheck(n,iternum=10000))
        return

    def gilbreathCheck(self):
        """
        docstring for gilbreathCheck
        check Gilbreath's conjecture
        2,3,5,7,11,13...
        1,2,2,4,2...
        1,0,2,2...
        1,2,0...
        1,2...
        1...
        difference sequence (absolute difference) always starts 
        with 1
        """
        arr = []
        n   = 201
        for i in range(1,n):
            k = sympy.prime(i)
            print(i,k)
            arr.append(k)

        arr = np.array(arr)
        for i in range(n//2):
            arr = np.abs(arr[1:] - arr[:-1])
            print(i,arr[:10])

        return

    def getEulerPhi(self,n):
        """
        docstring for getEulerPhi
        get the Euler's phi function
        n = \prod pi^ai
        return:
            res = n*\prod(1-1/pi)
        """
        factors = sympy.factorint(n)
        res = n 
        for i in factors:
            res = (res // i)*(i-1)
        return res

    def getAllFactors(self,n):
        """
        docstring for getAllFactors
        18 => [1,2,3,6,9,18]
        n:
            integer
        return:
            factor array
        """
        factors = sympy.factorint(n)
        indeces = []
        keys    = []
        res     = []
        for key in factors:
            value = factors[key]
            indeces.append(np.arange(value+1))
            # print(value)
            keys.append(key)

        combinations = self.getAllCombinator(indeces)
        # print(combinations)
        for line in combinations:
            num = 1 
            for i in range(len(line)):
                num *= (keys[i]**line[i])
            res.append(num)
        res.sort()

        return res

    def getRidTwoFive(self,n):
        """
        docstring for getRidTwoFive
        get rid of all twos and fives
        210 => 21
        """
        while n%2 == 0:
            n = n//2 
        while n%5 == 0:
            n = n//5
        return n
    def getDecimalLength(self,n):
        """
        docstring for getDecimalLength
        n:
            integer
        return:
            n is prime, return n - 1
            n is not prime, return the smallest factor k
            where 10^k = 1 mod n
        """
        if sympy.isprime(n):
            res = n - 1 
        else:
            n   = self.getRidTwoFive(n)
            phi = self.getEulerPhi(n)
            # print(phi)
            factors = self.getAllFactors(phi)
            # print(factors)
            res = 1
            for i in factors:
                k = self.getModN(10,i,n)
                if k == 1:
                    res = i
                    break


        return res 

    def getPascalTriangle(self,n):
        """
        docstring for getPascalTriangle
        n:
            integer
        return:
            array, 3 => [1,1,1,1,2,1]
        """
        assert n > 2 
        res = [1]
        for i in range(2,n+1):
            res.append(1)
            arr = []
            for j in range(1,i-1):
                # print(res[-i],res[1-i],res)
                k = res[-i-1+j] + res[-i+j]
                arr.append(k)
            res = res + arr
            res.append(1) 
            # print(res[-i:])

        # statistics 
        print("statistis begins")
        stati = {}
        for j,i in enumerate(res):
            if i == 3003:
                print((j*2)**0.5,j,i)
            if i in stati:
                stati[i] += 1 
            else:
                stati[i] = 1
        keys = []
        for i in stati:
            value = stati[i]
            # print(i,value)
            keys.append([value,i])
        keys.sort()
        print(keys[-10:])
        return res

    def testSummation(self):
        """
        docstring for testSummation
        """
        i,n,p = symbols("i n p")
        s = sympy.summation(i*p**(n-i),(i,1,n))
        # print(s)
        # print(s[1])
        s = -(n*p - n - p*p**n + p)/(p - 1)**2
        print(latex(s))

        s = (p**(n+1) - p)/(p-1)
        s = s.diff(p)*p 
        s = s.expand().simplify()
        print(latex(s))
        s = sympy.summation(i*p**i,(i,1,n))
        print(s)
        s = p*(n*p*p**n - n*p**n - p**n + 1)/(p - 1)**2 
        print(latex(s))
        return

    def getTupleNum(self,arr,d):
        """
        docstring for getTupleNum
        when d = 4
        abcd - dcba = (a-d)*999 + 90*(b-c)
        a > b > c > d
        """
        num = 0
        for i in range(len(arr)):
            k    = 10**(d-1-i)-10**(i)
            num += k*arr[i]
        return num

    def getDigitBlackHole(self,arr,d=4):
        """
        docstring for get4DigitBlackHole
        arr:
            1D array, [m,n]
            d = 4, num = 999m+90n
            d = 5, num = 9999m+990n
        """
        res = [arr]
        for k in range(100):
            num = self.getTupleNum(arr,d)
            digits = self.num2Digits(num)
            # print(k,num,digits,arr)
            digits.sort()
            arr = [0]*len(arr)
            if len(digits) == (d-1):
                arr[0] = digits[-1]
                for i in range(1,len(arr)):
                    arr[i] = digits[-i-2] - digits[i]
            else:
                for i in range(len(arr)):
                    arr[i] = digits[-i-1] - digits[i]
            # print(arr)
            if arr in res:
                res.append(arr)
                # print("res ",res)
                break 
            res.append(arr)
                
        return res 

    def getRidRepeat2D(self,arr):
        """
        docstring for getRidRepeat2D
        arr:
            2D array,
            [[2,3],[3,2],[10]] => [[2,3],[10]]
        """
        res = []
        for line in arr:
            count = 1
            for item in res:
                if line[0] in item:
                    count = 0
            if count == 1:
                res.append(line)
                    
        return res
        
    def numberBlackHole(self,d=5):
        """
        docstring for numberBlackHole
        """
        # d = 5
        n = d // 2
        arr = np.arange(9+n)
        combinations = itertools.combinations(arr,n)

        total = []
        recordCycle = []
        cycles = []
        for line in combinations:
            arr = []
            for i in range(n):
                j = line[-i-1] + i - n + 1 
                arr.append(j)
            
            if sum(arr) == 0:
                continue 
            if arr not in total:
                res = self.getDigitBlackHole(arr,d=d)
                final = res[-1]
                if final not in recordCycle:
                    recordCycle.append(final)
                    k = res.index(final)
                    cycles.append(res[k:-1])
            total += res
        # print(recordCycle)
        cycles = self.getRidRepeat2D(cycles)
        # print(cycles)
        for line in cycles:
            # print(line)
            res  = []
            Line = []
            for arr in line:
                num = self.getTupleNum(arr,d)
                res.append(num)
                Line.append(tuple(arr))
            print(res,Line)
            # print()
                

        print("----")
        # print(total)
            
        return

    def quadraticCurve(self):
        """
        docstring for quadraticCurve
        """
        theta,a,b,c = symbols("theta a b c")
        R = Matrix([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
        S = Matrix([[a,b/2],[b/2,c]])

        s = R.transpose()*S*R 
        s = s.expand()
        for i in range(2):
            for j in range(2):
                res = s[i,j].trigsimp()
                type1 = "{\\left(\\theta \\right)}"
                res = latex(res).replace(type1,"\\theta")
                
                print(res + "\\\\")
        return

    def testCombinations(self,n=5):
        """
        docstring for testCombinations
        """
        arr = np.arange(n+10-1)
        combinations = itertools.combinations(arr,n)

        count = 0
        for line in combinations:
            count += 1
            arr = []
            for i in range(n):
                j = line[-i-1] + i - n + 1 
                arr.append(j)
            # print(arr)

        print(count)

        x = self.xyz[0]
        a = Symbol("a")
        y = x**3-a*x**2+2*x-2*a
        print(y.factor())
        return

    def testNumberBlackHole(self,n):
        """
        docstring for testNumberBlackHole
        """
        res = [n]
        for i in range(100):
            digits = self.num2Digits(res[-1])
            digits.sort()
            num1 = 0
            num2 = 0
            for j in range(len(digits)):
                num1 = 10*num1 + digits[-j-1]
                num2 = 10*num2 + digits[j]
            num = num1-num2
            if num in res:
                res.append(num)
                break 
            res.append(num)
        print(res)
        print(len(res))
        return

    def testNumberCycle(self):
        """
        docstring for testNumberCycle
        """
        m = 1
        n = 10**m - 1 
        num = 0 
        for i in range(1,n):
            num = (n+1)*num + i 
        num += 1 
        for i in range(1,10):
            print(i*num, "\\\\")
        return

    def getNumberCountCycle(self,arr):
        """
        docstring for getNumberCountCycle
        [2,0,3]:
            [1,2,3] => [2,1,3]
        [2,0,4]
            [1,2,3,4] => [2,4,3,1]
        """
        m, k, n = arr
        input_arr = arr
        arr = np.arange(1,n+1)
        output = []

        count = 0 
        while len(arr) > 0:
            res = []
            for i in arr:
                count += 1 
                if count % m == k:
                    output.append(i)
                else:
                    res.append(i)
            arr = res
        print(input_arr,"output",output[-1])
                
        return output[-1]

    def testNumberCount(self):
        """
        docstring for testNumberCount
        """
        arr1 = []
        arr2 = []
        arr3 = []
        for i in range(1,10):
            res = self.getNumberCountCycle([2,0,i])
            if res == i:
                arr1.append(i)
            elif res == i - 1:
                arr2.append(i)
            elif res == 1:
                arr3.append(i)
        print(arr1,arr2)
        print(arr3)
        return

    def get9DigitNum(self):
        """
        docstring for get9DigitNum
        """
        
        A = np.arange(1,10)
        multiTable = []
        A1 = itertools.product(A,repeat=2)
        for line in A1:
            a = line[0]*line[1]
            multiTable.append(a)
        print(multiTable)
                

        A1 = itertools.permutations(A,9)
        count1 = 0 
        for line in A1:
            count1 += 1 
            if count1 == 1:
                print(line)
            num = 0 
            count = 0
            for i in range(9):
                num = 10*num + line[i]
                if num%(i+1) == 0:
                    count += 1 
            if count == 9:
                print(num)

            count = 0 
            for i in range(8):
                a = 10*line[i]+line[i+1]
                if a in multiTable:
                    count += 1
            if count == 8:
                print(num) 

            count = 0
            a = line[0]*line[1]*line[2]
            for i in range(3):
                j = 3*i 
                b = line[j]*line[j+1]*line[j+2]
                c = line[i]*line[i+3]*line[i+6]
                if b == a and c == a:
                    count += 1 
            if count == 3:
                print(num,line) 

        print(count1)

            
        return

    def getNumTwoInteger(self,arr):
        """
        docstring for getNumTwoInteger
        arr:
            1D array with length 2
            [a,b] => a+b,a*b,a-b,b-a,a/b,b/a
        """
        # a,b = Integer(arr[0]),Integer(arr[1]) 
        a,b = arr
        res = []
        res.append(a+b)
        res.append(a*b)
        res.append(a-b)
        res.append(b-a)
        symbols = ["+","*","-","--"]
        if b != 0:
            res.append(a/b)
            symbols.append("/")
        if a != 0:
            res.append(b/a)
            symbols.append("//")
        
        dicts = {}
        for key,value in zip(symbols,res):
            dicts[key] = value
        
        return dicts
    def getNumber24(self,arr):
        """
        docstring for getNumber24
        arr:
            1D array, four integers
        """
        # print(arr)
        result = False

        answers = []
        operations = []
        for i in range(4):
            for j in range(i+1,4):
                # get the answers
                line = arr.copy()
                a,b = arr[i],arr[j]
                line.remove(a)
                line.remove(b)
                line1 = [] 
                line1.append(Integer(line[0]))
                line1.append(Integer(line[1]))
                line = line1
                res = [Integer(a),Integer(b)]
                res = self.getNumTwoInteger(res)
                # print(res)
                # print(res,line)
                # condition 2 and 2 
                line2 = self.getNumTwoInteger(line)
                for key1 in res:
                    for key2 in line2:
                        ii = res[key1]
                        jj = line2[key2]
                        output = self.getNumTwoInteger([ii,jj])
                        for key in output:
                            # answers.append(output[key])
                            if output[key] == 24:
                                answers.append(24)                     
                                format_type = "(%s %s %s)%s(%s %s %s)"
                                data = (latex(a),key1,latex(b),key,latex(line[0]),key2,latex(line[1]))
                                if key1 in ["//","--"]:
                                    data = (latex(b),key1[0],latex(a),key,latex(line[0]),key2,latex(line[1]))
                                if key in ["//","--"]:
                                    data = (latex(line[0]),key2,latex(line[1]),key[0],latex(a),key1,latex(b))
                                if key2 in ["//","--"]:
                                    data = (latex(a),key1,latex(b),key,latex(line[1]),key2,latex(line[0]))
                                operation = format_type % data
                                if operation not in operations:
                                    operations.append(operation)
                # condition 2,1,1
                for key1 in res:
                    for k in range(2):     
                        ii = res[key1]
                        jj = line[k]
                        kk = line[1-k]
                        output = self.getNumTwoInteger([ii,jj])
                        for key2 in output:
                            ii = output[key2]
                            output2 = self.getNumTwoInteger([ii,kk])
                            for key in output2:
                                if output2[key] == 24:
                                    answers.append(24)                     
                                    format_type = "((%s %s %s)%s %s) %s %s"
                                    data = (latex(a),key1,latex(b),key2,latex(jj),key,latex(kk))
                                    if key1 in ["//","--"]:
                                        data = (latex(b),key1[0],latex(a),key2,latex(jj),key,latex(kk))
                                        
                                     
                                    if key2 in ["//","--"]:
                                        format_type = "(%s %s (%s %s %s)) %s %s"
                                        data = (latex(jj),key2[0],latex(a),key1,latex(b),key,latex(kk))
                                    if key in ["//","--"]:
                                        format_type = "%s %s ((%s %s %s) %s %s)"
                                        data = (latex(kk),key[0],latex(b),key1,latex(a),key2,latex(jj))
                                    operation = format_type % data
                                    if operation not in operations:
                                        operations.append(operation)
                                        # print(ii,jj,kk)
                                
                if 24 in answers:
                    # print("yes")
                    result = True
                    index = answers.index(24)
                    # print(operations[index])
                # print(answers)
                    

        # print(answers)

        return result,operations
    def game24(self):
        """
        docstring for game24
        24 game, a kind of number's game
        try to get 24 with five operations when 4 integers 
        were given
        """

        for i in range(22,28):
            # print(i)
            arr = [i]*4 
            res,operations = self.getNumber24(arr)
            if res:
                print(arr,operations)
                # print(arr)
            # print(arr,res)

        n = 2
        arr = np.arange(1,22+n)
        combinations = itertools.combinations(arr,n)

        """
        count = 0
        for line in combinations:
            count += 1
            # arr = []
            # for i in range(n):
            #     j = line[-i-1] + i - n + 1 
            #     arr.append(j)
            # item = [arr[0]]*3+[arr[1]]
            arr = line 
            item = [arr[0]]*3+[arr[1]]
            res,operations = self.getNumber24(item)
            if res:
                # print(res,item,operations[0])
                print(item,operations[0])
            item = [arr[1]]*3+[arr[0]]
            res,operations = self.getNumber24(item)
            if res:
                # print(res,item,operations[0])
                print(item,operations[0])

            item = [arr[1]]*2+[arr[0]]*2
            res,operations = self.getNumber24(item)
            if res:
                print(item,operations[0])
        """

        n = 4
        arr = np.arange(1,24+n)
        combinations = itertools.combinations(arr,n)

        count = 0
        for line in combinations:
            count += 1 
            arr = []
            for i in range(n):
                j = line[-i-1] + i - n + 1 
                arr.append(j)
            item = arr
            res,operations = self.getNumber24(item)
            if res and len(operations) < 3 and max(item) > 13:
                # print(res,item,operations[0])
                if operations[0].count("/") > 2:
                    print(item,operations)
        print(count)
        return


    def divisibility(self):
        """
        docstring for divisibility
        """
        arr = []
        for i in range(-500,1000):
            if i == 3:
                continue 

            a = i - 3 
            b = i**3 - 3 
            if b%a == 0:
                print(i,a,b)
                arr.append(i)
        print(arr)
        for i in range(1,13):
            print((i*i)%13)


        for i in range(1,10):
            p = sympy.prime(i)
            print(i,p)
            n = sqrt((p**5-1)/(p-1))
            if n.is_integer:
                print(p,n,n**2,p**4)
                break
        n = 2**(2**2)+1 
        print(n)
        print(self.getModN(2,n-1,p))
        # print(sympy.factorint(n))


        x,y = self.xyz[:2]
        # n = Integer(3) 
        n = Symbol("n")
        a = n
        b = 3*n*(n+1)/2 
        c = b*(2*n+1)/3
        d = (b/3)**2 
        s = (a*x**3+b*x**2+c*x+d)
        print(factor(s))
        print(latex(factor(s)))
        s = s.subs(x,-1-n/2).expand()
        print("after substitution",s)

        s = (a*x**3+b*x**2+c*x+d) - (x+n+1)**3 
        print(factor(s))
        
        S1 = n*(n**2 + 2*n*x + n + 2*x**2 + 2*x)/2
        S2 = (n + 2*x + 1)/2 
        s = S1 - S2**2 
        print("S1 = S2**2")
        s1 = factor(s)
        print(latex(s1))

        s = s.expand().collect(x)
        print(latex(s))


        # for i in range(10000):
        #     v = Integer(i**3-16*i+16)
        #     u = sqrt(v)
        #     if u.is_integer:
        #         print(i,u,v)
        x = np.arange(1,100000)
        u = x**3-16*x+16 
        return

    def continuedFraction(self):
        """
        docstring for continuedFraction
        """
        m,n = symbols("m n")
        s   = 4*m*n+1
        k   = n + m*s  
        D   = k**2 + 4*m*n+1
        res   = D - (k-4*n)**2  
        print(factor(res))
        res   = D - (s*(2*m+1)+2*n-k)**2
        # res   = D - (s*(2*m)-k)**2
        print(factor(res))

        m,n = 3,3
        s   = 4*m*n+1
        k   = n + m*s  
        D   = k**2 + 4*m*n+1
        print(m,n,k,D,self.getContinueSeq(D))
        return
    
    def solvePuzzles(self):
            """
            docstring for solvePuzzles
            """
            a = sqrt(3)
            p = self.getCubicSol([1,a,3,a])
            print(latex(p))

            x = self.getCubicSol([3,3,3,1])
            print(x)

            x = -Integer(1)/3 + 2**(1/3)/3 - 4**(1/3)/3
            y = x*(x*(x*3+3)+3)
            print(y.expand())

            return  

    def progression(self):
        """
        docstring for progression
        S_{n+1}&=&S_{n}+\frac{1}{2}S_{n-1}+\frac{1}{6}S_{n-2}

         1  
         0  1  2  2  2  3  3  
         2  4  
         3  4  5  5  5  6  6  
         4  7  
         6  7  8  8  8  9  9  
         8 10  
         4 10 11 11 11 12 12 
        11 13 
        12 13 14 14 14 15 15 
        13 16 
        15 16 17 17 17 18 18 
        17 19 
        17 19 20 20 20 21 21 
        20 22 
        21 22 23 23 23 24 24 
        20 25 
        24 25 26 26 26 27 27 
        26 28 
        26 28 29 29 29 30 30 
        29 31 
        30 31 32 32 32 33 33 
        31 34

        """
        arr = [1,2,3]
        p   = [1,Integer(1)/2,Integer(1)/6]
        for i in range(5):
            num = 0 
            for j in range(3):
                num += p[j]*arr[-j-1]
            arr.append(num)
            if i == 1:
                continue
            a,b = sympy.fraction(num)
            factors = sympy.factorint(b)
            values = []
            for key in factors:
                values.append(factors[key])
            # print(values)
            # print(i+4,values)
            a = values[1] 
            print("%2d"%(a))

        return 

    def getCountDict(self,arr):
        """
        docstring for getCountDict
        arr:
            1d array
            [1,1,1,2,2,3] => {1:3,2:2,3:1}
        """
        res = {}
        for i in arr:
            if i in res:
                res[i] += 1 
            else:
                res[i] = 1 
        return res 
    def testCubicContinuedFrac(self):
        """
        docstring for testCubicContinuedFrac
        """
        # x = self.getContinueSeq(2 , n=3, target=0)
        # print(x)
        t1 = time.time()
        x = self.getCubicContinueSeq(4,count=8000)
        # print(x)
        t2 = time.time()
        print("time",t2 - t1)

        res = self.getCountDict(x)
        print(res)
        x = np.array(x)
        indeces = np.arange(len(x))
        plt.plot(indeces,x)
        plt.show()
        x = sum(np.log(x))/len(x)
        x = np.exp(x)
        print("Khinchin's constant:",x)
       
        return

    def khinchin(self):
        """
        docstring for khinchin
        """
        r = np.arange(1,100000)
        x = np.log((1+1/r/(r+2)))*(np.log(r)/np.log(2))
        x = np.exp(np.sum(x))
        print(x)

        res = Decimal(1)
        for i in r:
            a = Decimal(1+1/i/(i+2))
            # a = np.log(a)
            b = Decimal(np.log(i)/np.log(2))
            res = res*(a**b)
        print("khinchin's constant",res)
        return

    def getConwayData(self):
        """
        docstring for getConwayData
        """
        data = [[4 ,   [63]],
                [7 ,   [64,62]],
                [12 ,  [65]],
                [12 ,  [66]],
                [4 ,   [68]],
                [5 ,   [69]],
                [12 ,  [84,55]],
                [6 ,   [70]],
                [8 ,   [71]],
                [10 ,  [76]],
                [10 ,  [77]],
                [14 ,  [82]],
                [12 ,  [78]],
                [14 ,  [79]],
                [18 ,  [80]],
                [42 ,  [81,29,91]],
                [42 ,  [81,29,90]],
                [26 ,  [81,30]],
                [14 ,  [75,29,92]],
                [28 ,  [75,32]],
                [14 ,  [72]],
                [24 ,  [73]],
                [24 ,  [74]],
                [5 ,   [83]],
                [7 ,   [86]],
                [10 ,  [87]],
                [10 ,  [88]],
                [8 ,   [89,92]],
                [2 ,   [1]],
                [9 ,   [3]],
                [9 ,   [4]],
                [23 ,  [2,61,29,85]],
                [2 ,   [5]],
                [6 ,   [28]],
                [32 ,  [24,33,61,29,91]],
                [32 ,  [24,33,61,29,90]],
                [8 ,   [7]],
                [3 ,   [8]],
                [5 ,   [9]],
                [6 ,   [10]],
                [10 ,  [21]],
                [18 ,  [22]],
                [18 ,  [23]],
                [6 ,   [11]],
                [10 ,  [19]],
                [8 ,   [12]],
                [7 ,   [13]],
                [8 ,   [14]],
                [12 ,  [15]],
                [20 ,  [18]],
                [34 ,  [16]],
                [34 ,  [17]],
                [20 ,  [20]],
                [10 ,  [6,61,29,92]],
                [7 ,   [26]],
                [7 ,   [27]],
                [11 ,  [25,29,92]],
                [13 ,  [25,29,67]],
                [21 ,  [25,29,85]],
                [17 ,  [25,29,68,61,29,89]],
                [2 ,   [61]],
                [1 ,   [33]],
                [4 ,   [40]],
                [7 ,   [41]],
                [14 ,  [42]],
                [14 ,  [43]],
                [7 ,   [38,39]],
                [4 ,   [44]],
                [6 ,   [48]],
                [8 ,   [54]],
                [10 ,  [49]],
                [16 ,  [50]],
                [28 ,  [51]],
                [28 ,  [52]],
                [9 ,   [47,38]],
                [12 ,  [47,55]],
                [12 ,  [47,56]],
                [16 ,  [47,57]],
                [18 ,  [47,58]],
                [24 ,  [47,59]],
                [23 ,  [47,60]],
                [16 ,  [47,33,61,29,92]],
                [6 ,   [45]],
                [5 ,   [46]],
                [15 ,  [53]],
                [6 ,   [38,29,89]],
                [10 ,  [38,30]],
                [10 ,  [38,31]],
                [3 ,   [34]],
                [27 ,  [36]],
                [27 ,  [35]],
                [5 ,   [37]]]

        return data 

    def getConwayStrings(self):
        """
        docstring for getConwayStrings
        """
        strings = [ "1112",
                    "1112133",
                    "111213322112",
                    "111213322113",
                    "1113",
                    "11131",
                    "111311222112",
                    "111312",
                    "11131221",
                    "1113122112",
                    "1113122113",
                    "11131221131112",
                    "111312211312",
                    "11131221131211",
                    "111312211312113211",
                    "111312211312113221133211322112211213322112",
                    "111312211312113221133211322112211213322113",
                    "11131221131211322113322112",
                    "11131221133112",
                    "1113122113322113111221131221",
                    "11131221222112",
                    "111312212221121123222112",
                    "111312212221121123222113",
                    "11132",
                    "1113222",
                    "1113222112",
                    "1113222113",
                    "11133112",
                    "12",
                    "123222112",
                    "123222113",
                    "12322211331222113112211",
                    "13",
                    "131112",
                    "13112221133211322112211213322112",
                    "13112221133211322112211213322113",
                    "13122112",
                    "132",
                    "13211",
                    "132112",
                    "1321122112",
                    "132112211213322112",
                    "132112211213322113",
                    "132113",
                    "1321131112",
                    "13211312",
                    "1321132",
                    "13211321",
                    "132113212221",
                    "13211321222113222112",
                    "1321132122211322212221121123222112",
                    "1321132122211322212221121123222113",
                    "13211322211312113211",
                    "1321133112",
                    "1322112",
                    "1322113",
                    "13221133112",
                    "1322113312211",
                    "132211331222113112211",
                    "13221133122211332",
                    "22",
                    "3",
                    "3112",
                    "3112112",
                    "31121123222112",
                    "31121123222113",
                    "3112221",
                    "3113",
                    "311311",
                    "31131112",
                    "3113112211",
                    "3113112211322112",
                    "3113112211322112211213322112",
                    "3113112211322112211213322113",
                    "311311222",
                    "311311222112",
                    "311311222113",
                    "3113112221131112",
                    "311311222113111221",
                    "311311222113111221131221",
                    "31131122211311122113222",
                    "3113112221133112",
                    "311312",
                    "31132",
                    "311322113212221",
                    "311332",
                    "3113322112",
                    "3113322113",
                    "312",
                    "312211322212221121123222113",
                    "312211322212221121123222112",
                    "32112"]

        return strings

    def lookSay(self,look_s):
        """
        '111222112' => '31322112'
        """
        st = ''
        for s in re.finditer(r"(\d)\1*",look_s):
            # print(s,s.group(0),s.group(1))
            st = st + str(len(s.group(0)))+s.group(1)
        return st
    def getPolymonialValues(self,p,x):
        """
        docstring for getPolymonialValues
        p:
            1d array, coefficients of the polynomial 
            length n+1, n is the order 
            of the polynomial
        x:
            numerial value or symbolic value
        """
        res = 0 
        for i in p:
            res = res*x + i
        
        return res

    def checkConwayData(self,data,strings):
        """
        docstring for checkConwayData
        """
        count = 0 
        for i,line in enumerate(data):
            # print(i+1,line[0],len(strings[i]),strings[i])
            nextIterCh = self.lookSay(strings[i])
            arr = ""
            for j in line[1]:
                j = j - 1 
                arr += strings[j]
            # print(i+1,strings[i],nextIterCh,arr)
            if nextIterCh == arr:
                count += 1 
            else:
                print(i+1,line,strings[i],nextIterCh,arr)

        print("equal numbers: ",count)

        arr = "1"
        for i in range(100):
            a   = len(arr)
            arr = self.lookSay(arr)
            print(i,len(arr)/a)

        return
    def conwayConstant(self):
        """
        docstring for conwayConstant
        look-and-say sequence and Conway Constant

        """
        
        data = self.getConwayData()
        strings = self.getConwayStrings() 
        # check the data and strings
        # self.checkConwayData(data,strings)
        
        conwayAtoms = sympy.zeros(92)
        count = 0 
        for i,line in enumerate(data):
            # print(i+1,line[0],len(strings[i]),strings[i])
            if line[0] == len(strings[i]):
                count += 1 
            for j in line[1]:
                j = j - 1 
                b = data[j][0]
                a = data[i][0]
                conwayAtoms[j,i] = Integer(b)/a
        print("equal length numbers: ",count)

        # 1, 11, 21, 1211, 111221, 312211, 13112221, 1113213211
        vector = [0]*92 
        vector[23] = 5
        vector[38] = 5
        vector = Matrix(vector)
        print(vector)
        # print(vector)
        a = 1 
        b = 1
        for i in range(100):
            vector = conwayAtoms*vector 
            b = sum(vector)
            print(i,b)
            if i%10 == 0:
                print(i,float(b/a),vector)
            a = b
        x = self.xyz[0]
        for i in range(92):
            conwayAtoms[i,i] = -x 
        # t1 = time.time()
        # s = conwayAtoms.det()
        # t2 = time.time()
        # print("time:",t2 - t1)
        # s = s.factor()
        # print(s)
        # print(latex(s))

        """
        
        x^{19} \left(x - 1\right) \left(x + 1\right)^{2} \left(x^{70} - 
        x^{69} - 2 x^{67} + x^{66} + x^{65} + x^{64} - x^{62} - x^{60} - 
        x^{58} + 3 x^{57} + 2 x^{56} + x^{55} - 3 x^{54} - 7 x^{53} + 
        4 x^{52} - 5 x^{51} + 11 x^{50} - 6 x^{49} + 5 x^{48} + 3 x^{47} - 
        4 x^{46} - x^{45} - 5 x^{44} - 4 x^{43} + 12 x^{42} - 6 x^{41} + 
        12 x^{40} - 17 x^{39} + 7 x^{38} + 2 x^{37} - 6 x^{36} + 9 x^{35} - 
        13 x^{34} + 9 x^{33} + 2 x^{32} + 4 x^{31} - 7 x^{30} + 3 x^{29} - 
        14 x^{28} + 11 x^{27} - 8 x^{26} + 15 x^{25} - 20 x^{24} + 
        33 x^{23} - 39 x^{22} + 43 x^{21} - 51 x^{20} + 56 x^{19} - 
        52 x^{18} + 49 x^{17} - 59 x^{16} + 54 x^{15} - 45 x^{14} + 
        54 x^{13} - 53 x^{12} + 39 x^{11} - 42 x^{10} + 43 x^{9} - 
        38 x^{8} + 37 x^{7} - 39 x^{6} + 33 x^{5} - 25 x^{4} + 
        23 x^{3} - 13 x^{2} + 8 x - 6\right)
        """

        # x = 1.3029910351028386
        # polynomial of 70 order
        x = 1.3029910351028386
        p = [1,-1,0,-2,1,1,1,0,-1,0,-1,0,-1,3,2,1,-3,-7,
             4,-5,11,-6,5,3,-4,-1,-5,-4,12,-6,12,
             -17,7,2,-6,9,-13,9,2,4,-7,3,-14,11,
             -8,15,-20,33,-39,43,-51,56,-52,49,
             -59,54,-45,54,-53,39,-42,43,-38,37,
             -39,33,-25,23,-13,8,-6]
        print(len(p),p)
        print(self.getPolymonialValues(p,x))


        # p = [1,0,-1,-2,-1,2,2,1,-1,-1,-1,-1,-1,2,5,
        #      3,-2,-10,-3,-2,6,6,1,9,-3,-7,-8,-8,10,
        #      6,8,-5,-12,7,-7,7,-1,-3,10,1,-6,-2,-10,
        #      -3,2,9,-3,14,-8,0,-7,9,3,-4,-10,-7,12,7,
        #      2,-12,-4,-2,5,0,1,-7,7,-4,12,-6,3,-6]

        p = [1,0,-1,-2,-1,2,2,1,-1,-1,-1,-1,-1,2,5,
             3,-2,-10,-3,-2,6,6,1,9,-3,-7,-8,-8,10,
             6,8,-5,-12,7,-7,7,1,-3,10,1,-6,-2,-10,
             -3,2,9,-3,14,-8,0,-7,9,3,-4,-10,-7,12,7,
             2,-12,-4,-2,5,0,1,-7,7,-4,12,-6,3,-6]

        # x = Decimal(1.3035772690342963912570991121525518907307025046594)
        x = Decimal(1.303577269034296391257099)
        print(self.getPolymonialValues(p,x))


    
        return


    def generalFibonacci(self):
        """
        docstring for generalFibonacci
        """
        n = 6
        arr = [1]*n 
        for i in range(n-1):
            num = sum(arr[-n:])
            arr.append(num)
        print(arr)

        A = sympy.zeros(n)
        for i in range(n):
            for j in range(n):
                A[i,j] = arr[n-1-i+j]
        print(A,A.det())
        # 1,1,...,1,m,2m-1,4m-3,8m-7,...,2^{m-1}\times(m-1)+1
        return  

    def testCubicSum(self):
        """
        docstring for testCubicSum
        1 - 1/3^3 + 5^3 + ...
        """
        
        res = 0 
        pos = 1
        for i in range(10000):
            res += pos/((2*i+1)**3)
            pos  = -pos
        print(res)
        print(np.pi**3/32)
        return

    def mersennePrimes(self):
        """
        docstring for mersennePrimes
        [2, 3, 5, 7, 13, 17, 19, 31, 61, 
         89, 107, 127, 521, 607, 1279, 
         2203, 2281, 3217, 4253, 4423]
        """
        
        Mprime = []
        for i in range(1,1000):
            if i % 10 == 0:
                print(i)
            p = sympy.prime(i)
            Mp = 2**p - 1 
            if sympy.isprime(Mp):
                print(i,p)
            # L = 4 
            # for j in range(p-2):
            #     L = (L**2 - 2) % Mp
            # if L%Mp == 0:
            #     Mprime.append(p)
            #     print("M_%d is a prime"%(p))

        print(Mprime)
        return
    def test(self):
        """
        docstring for test
        """    
        
        # self.modularEquation()
        # self.weierstrassForm()
        # print(self.getFactors(1459))
        # self.getWeierstrassForm()
        # self.conicSection()
        # t1 = time.time()
        # for i in range(1,11):
            
        #     # m = sympy.prime(i)
        #     # print("prime",m)
        #     self.testABCElliptic(10000,i)
        #     t2 = time.time()
        #     print("time:",t2 - t1)
        #     t1 = t1
        # self.testABC()
        # self.getGeneralPellSol()
        # self.getABCByNM(2,4)
        # self.generalConic()
        # self.conicProp()
        # self.quadraticResidue()
        # self.getQradraticResByReci(2017,5003)
        # self.huiwenTest()
        # self.gilbreathCheck()
        # print(self.getDecimalLength(998001))
        # print((10**588 -1 )//343)
        # self.getPascalTriangle(100)
        # print(self.getCombinator(15,5))
        # self.testSummation()
        # for d in range(3,6):
        #     print("d = ",d)
        #     self.numberBlackHole(d=d)
        # self.quadraticCurve()
        # self.testCombinations(n=10)
        # self.testNumberBlackHole(865296432)
        # self.testNumberCycle()
        # self.testNumberCount()
        # self.get9DigitNum()
        # self.game24()
        # self.divisibility()
        # self.continuedFraction()
        # self.solvePuzzles()
        # self.progression()
        # self.testCubicContinuedFrac()
        # self.khinchin()
        # self.conwayConstant()
        # self.generalFibonacci()
        # self.testCubicSum()
        # self.mersennePrimes()


        return

formula = Formulas()
formula.test() 