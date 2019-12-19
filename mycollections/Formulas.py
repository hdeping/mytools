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
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import json



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
                factors.append(i)
                factors.append(m // i)

        if count == 0:
            print(m,"is prime")
        else:
            print(factors)
        return factors

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
        # self.remainderTheorem()  
        # self.pellSol()
        # self.getInitPell(999)
        # self.pellSol(31)
        # num = 277777788888899
        # digits = self.num2Digits(num)
        # print(digits)
        # self.getMultiNum(num)

        # self.remainderTheorem()
        # self.pythagorean()

        # self.testInverseNum()
        self.testPrime()
        # self.isPrime(39252)
        # print(self.getFactors(39252))
        # self.testAverageProblem()
        
        return
  

formula = Formulas()
# formula.diophantine() 
# formula.bernoulliGen() 
formula.test() 
# formula.dealData() 
# formula.continueFrac()   
