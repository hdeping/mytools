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

class Formulas():
    """Formulas Deriations with sympy"""
    def __init__(self):
        super(Formulas, self).__init__()
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
    def test(self):
        """
        docstring for test
        """
        print(self.getBinomial(3))
        print(self.getBinomial(4))
        print(self.getBinomial(6))
        
        return
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
  

formula = Formulas()
formula.bernoulliNum2() 
# formula.bernoulliGen() 
# formula.test()     
