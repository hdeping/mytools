#!/usr/bin/env python
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
import numpy as np
from mytools import MyCommon


class Formulas(MyCommon):
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
    def residueTheorem(self):
        """
        docstring for residueTheorem
        check the residue theorem based 
        on the diophantine equation
        """
        p = np.array([55,73,111,128])
        P = np.prod(p)
        q = P // p
        a = [2,3,7,1]

        solution = 0
        for i in range(len(p)):
            value = self.diophantine(p[i],q[i])[1]
            solution += value*a[i]*q[i]
        print(solution)
        solution = solution % P
        print(solution)
        print(solution % p)
        

        return
    def pellSol(self):
        """
        docstring for pellSol
        x_{2}   &=& x_{1}^{2}+Dy_{1}^{2}\\
        x_{n+1} &=& 2x_{1}x_{n}-x_{n-1}\\
        y_{2}   &=& 2x_{1}y_{1}\\
        y_{n+1} &=& 2x_{1}y_{n}-y_{n-1}
        """
        x = [[1,0],[3,2]] # D = 2
        
        for i in range(10):
            x0 = 6*x[-1][0] - x[-2][0]
            x1 = 6*x[-1][1] - x[-2][1]
            print(x0,x1,x0**2 - 2*x1**2)
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

    def getInitPell(self,D):
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
    def test(self):
        """
        docstring for test
        """
        # print(self.getBinomial(3))
        # print(self.getBinomial(4))
        # print(self.getBinomial(6))
        # divisor = self.getDivisorSeq(39,37)
        # print(divisor)

        # self.diophantine(1027,712)
        # self.residueTheorem()  
        # self.pellSol()
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
  

formula = Formulas()
# formula.diophantine() 
# formula.bernoulliGen() 
formula.test()     
