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
from sympy import expand,simplify,cos,sin,exp,sqrt
from sympy import latex
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

        if count == 0:
            print(m,"is prime")
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
    def test(self):
        """
        docstring for test
        """
        # self.testFermat()
        # self.diophantine(98765432123456789,12345678987654321)
        # print(self.getFactors(98765432123456789))
        print(self.getFactors(12345678987654321))
        # self.isPrime
        print(self.fermatPrimeTest(987654321234567834349))
        # self.testCubic()
        # self.hardyWeinberg()
        # self.testAllMod()
        # self.selectNum70()
        # self.polygon17()
        # self.polygon257()

        # self.idCardCheck()
        # self.fermatAndGroup(56)
        
        return

  

formula = Formulas()

formula.test() 
  
