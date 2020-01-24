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
from sympy import Matrix,limit,tan,Integer
from sympy.solvers import diophantine
import numpy as np
from mytools import MyCommon
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import json
import time


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
    def pellSol(self,D):
        """
        docstring for pellSol
        x_{2}   &=& x_{1}^{2}+Dy_{1}^{2}\\
        x_{n+1} &=& 2x_{1}x_{n}-x_{n-1}\\
        y_{2}   &=& 2x_{1}y_{1}\\
        y_{n+1} &=& 2x_{1}y_{n}-y_{n-1}
        """
        a,b = self.getInitPell(D)
        x = [[1,0],[a,b]] # D = 2
        
        for i in range(10):
            x0 = 2*a*x[-1][0] - x[-2][0]
            x1 = 2*a*x[-1][1] - x[-2][1]
            print("|%d|%d|%d|%d|"%(i+2,x0,x1,x0**2 - D*x1**2))
            x.append([x0,x1])
        
        return
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

    def getContinueSeq(self,D):
        """
        docstring for getInitPell
        D:
            x^2 - Dy^2 = 1, 
            positive integer and not square one and  D > 1
            x_n = a_n + 1/(a_{n+1} + x_{n-1})
        """
        x = sympy.sqrt(D)
        # print(x)
        result = []
        target = 2*(x//1)
        for i in range(100):
            a = x // 1 
            x = sympy.simplify(1/(x - a))
            result.append(a)
            if a == target:
                break 

        return result
    def getInitPell(self,D):
        """
        docstring for getInitPell
        D:
            x^2 - Dy^2 = 1,
        """
        result = self.getContinueSeq(D)
        print("result",result)
        m      = result[0]
        sequence = result[1:]
        A = [1,0]
        B = [0,1]
        for i in range(60):
            item = sequence[i % len(sequence)]
            a = item*A[-1] + A[-2]
            b = item*B[-1] + B[-2]
            x = m*b + a
            y = b 
            judge = x**2 - D*y**2
            A.append(a)
            B.append(b)
            print(i+1,x,y,judge)
            if judge == 1:
                print(i+1,x,y,judge)
                break
        return x,y


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
        sn = ""
        number = ""
        for line in res:
            arr = []
            for i,item in enumerate(line):
                if item > 0:
                    arr += [i+1]*item
            item = "%d S_{%s} \\\\"
            ch   = ""
            for i in arr:
                ch += str(i)
            k = self.getGeneralCombinator(arr)
            item = item%(k,ch)
            number = "%s&%d"%(number,k)
            print(item)
            output.append(arr)
            sn = "%s %s +"%(sn,item)

        print(sn)
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

    def getSnByMat(self):
        """
        docstring for getSnByMat
        """
        n = 6
        a = sympy.symbols("a0:%d"%(n+1))
        res = self.getCombinatorEqnRecursive(n,n)
        print(res)
        res,sn = self.getAnotherCombinator(res)
        print(res)
        # print(self.getGeneralCombinator([4,3]))

        # get the matrix of polynomials 
        # A = XB ==> X = AB^{-1}
        polyA = []
        tmp   = []
        polyB = []
        N     = Symbol("n")
        for line in res:
            coef = {}
            for i in line:
                if i not in coef:
                    coef[i] = 1 
                else:
                    coef[i] += 1 
            num  = self.getGeneralCombinator(list(coef.keys()))
            item = num*self.getCombinator(N,len(line))
            polyA.append(item)
            item = 1
            for i in coef:
                num  = self.getCombinator(N,i)
                item = item*num**coef[i]
            tmp.append(item)
            print(coef)
        while len(tmp) > 0:
            polyB.append(tmp.pop())
        print("B",polyB)

        A = 
        return

    def getCoefMatrix(self,polynomials):
        """
        docstring for getCoefMatrix
        """

        N  = Symbol("n")
        n  = len(polynomial)
        # get a null matrix
        matrix = []
        for i in range(n):
            line = []
            for j in range(n):
                line.append(0)
            matrix.append(line)
        

        for line in polynomials:
            line = sympy.Poly(line,N).as_dict()
            print(line)
        return matrix
    def test(self):
        """
        docstring for test
        """
        
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
        self.getSnByMat()



        return

formula = Formulas()
formula.test() 