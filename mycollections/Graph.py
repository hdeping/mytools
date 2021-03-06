#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-09-02 10:19:19
    @project      : tools for graphs
    @version      : 1.0
    @source file  : Graph.py

============================
"""
from vpython import *
import numpy as np 
from PIL import Image
import matplotlib 
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import sympy
from tqdm import tqdm

class MyVpython():
    """docstring for MyVpython"""
    def __init__(self):
        super(MyVpython, self).__init__()

    def oneDMotion(self):
        """
        docstring for equiSpeedMotion
        """

        size = 0.1   # 木塊邊長
        L = 1        # 地板長度
        v = 0.03     # 木塊速度
        t = 0        # 時間
        dt = 0.01    # 時間間隔
        a  = 10   # accelerating rate


        scene = canvas(title="1D Motion", 
                       width=800, height=600, x=0, y=0, 
                       center=vec(0, 0.1, 0), 
                       background=vec(0, 0.6, 0.6))
        floor = box(pos=vec(0, 0, 0), 
                    size=vec(L, 0.1*size, 0.5*L), 
                    color=color.blue)
        cube = box(pos=vec(-0.5*L + 0.5*size, 0.55*size, 0), 
                   size=vec(size, size, size), 
                   color=color.red, v=vec(v, 0, 0))
        #cube = box(pos=vec(-0.5*L + 0.5*size, 0.55*size, 0), length=size, height=size, width=size, color=color.red, v=vec(v, 0, 0))
        gd = graph(title="x-t plot",  
                   width=600, height=450, x=0, y=600, 
                   xtitle="t(s)", ytitle="x(m)")
        gd2 = graph(title="v-t plot", width=600, 
                    height=450, x=0, y=1050, xtitle="t(s)", ytitle="v(m/s)")
        xt = gcurve(graph=gd, color=color.red)
        vt = gcurve(graph=gd2, color=color.red)


        while(cube.pos.x <= 0.5*L- 0.5*size):
            rate(1000)
            cube.pos.x += (v+dt*a)*dt
            xt.plot(pos = (t, cube.pos.x))
            vt.plot(pos = (t, cube.v.x))
            t += dt

        print("t = ", t)
        return

    def planetMotion(self):
        """
        docstring for planetMotion
        """
        size = 2E10            # 星球半徑, 放大約2000倍, 否則會看不見
        sun_m = 1988500E24     # 太陽質量
        d = 1.5E11             # 地球平均軌道半徑為 1.5E11 m
        v0 = 29780             # 地球公轉平均速率為 29780 m/s
        eps = 10000            # 計算週期用的精準度
        m = 4                  # 自訂行星 planet 軌道半徑為 m*d
        ec = 0.5               # 自訂行星軌道離心率(eccentricity), 若 m != 1 時設為 0
        G = 6.67408E-11        # 重力常數
        t = 0                  # 時間
        dt = 60*60*6            # 時間間隔

        """
         2. 畫面設定
            (1) 用 canvas 物件作為顯示動畫用的視窗 http://www.glowscript.org/docs/VPythonDocs/canvas.html
            (2) 用 sphere 物件產生星球 http://www.glowscript.org/docs/VPythonDocs/sphere.html
            (3) 星球的半徑要手動調整比例, 否則會看不到星球
        """

        title = "Kepler's Third Law of Planetary Motion"
        scene = canvas(title=title, 
                       width=900, height=900,
                       x=0, y=0, background=color.white)
        # 產生太陽 sun, 地球 earth 及自訂行星 planet
        sun = sphere(pos=vec(0,0,0), radius=size, 
                     m=sun_m, color=color.orange, emissive=True)
        earth = sphere(pos=vec(d, 0, 0), radius=size, 
                       texture=textures.earth, 
                       make_trail=True,
                       trail_color=color.blue, 
                       retain=365, v=vec(0, v0, 0))

        planet = sphere(pos=vec(m*(1+ec)*d, 0, 0), 
                        radius=size, color=color.red,
                        texture=textures.earth, 
                        make_trail=True,
                        retain=365*m, 
                        v=vec(0, v0/sqrt(m)*sqrt((1-ec)/(1+ec)), 0))
        line = cylinder(pos=vec(0, 0, 0), 
                        axis=vec(m*(1+ec)*d, 0, 0), 
                        radius=0.3*size, 
                        color=color.yellow)

        # 原來的寫法為 scene.lights = [local_light(pos = vec(0,0,0), 
        # color = color.white)]
        # 在 VPython 7 中 canvas.lights 無法設定為 local_light, 
        # 只能另外在太陽處放置另一個光源 lamp
        lamp = local_light(pos=vec(0,0,0), 
                           color=color.white)

        """
        3. 星球運動部分
        """
        while(True):
            rate(60*24)
            # 計算行星加速度、更新速度、位置
            earth.a = -G*sun.m / earth.pos.mag2 * earth.pos.norm()
            earth.v += earth.a*dt
            earth.pos += earth.v*dt
            planet.a = -G*sun.m / planet.pos.mag2 * planet.pos.norm()
            planet.v += planet.a*dt
            planet.pos += planet.v*dt
            # 判斷行星是否回到出發點
            if(abs(earth.pos.x - d) <= eps):
                print("t_Earth =", t)
            if(abs(planet.pos.x - m*(1+ec)*d) <= eps):
                print("t_planet =", t)
            # 更新時間
            t += dt

        return

    def spherePacking(self):
        """
        docstring for spherePacking
        """
        radius = 0.1
        dist   = radius*3
        spheres = []
        postions = []
        sizes = [600,600]
        n,m,l = 3,3,3
        scale = 3**0.5/2
        dt = 1 # degrees

        title = "sphere packing"
        for i in range(n):
            for j in range(m):
                for k in range(l):  
                    pos = vec(dist*(i/2+j),dist*scale*i,k*dist)
                    postions.append(pos)

        scene = canvas(title=title, 
                       width=sizes[0], 
                       height=sizes[1],
                       x=0, y=0, 
                       center = postions[(n*m*k)//2],
                       background=color.white)

        for pos in postions:
            print(pos)
            solid = sphere(pos=pos, radius=radius, 
                           trail_color=color.blue, 
                           texture=textures.earth,
                           emissive=True)
            spheres.append(solid)

        theta = dt*np.pi/180
        count = 0 
        while(1):
            rate(100)
            for solid in spheres:
                x = solid.pos.x
                z = solid.pos.z
                mat = [np.cos(theta),np.sin(theta)]
                solid.pos.x = mat[0]*x-mat[1]*z
                solid.pos.z = mat[1]*x+mat[0]*z
            count += 1 
            if count > 1:
                break

        return
    def test(self):
        """
        docstring for test
        """
        # self.oneDMotion()
        # self.planetMotion()
        self.spherePacking()
        return

class CellularAutomata():
    """docstring for CellularAutomata"""
    def __init__(self):
        super(CellularAutomata, self).__init__()

    def getRules(self,ruleType):
        """
        docstring for getRules
        ruleType:
            integer 
        """
        rules = []
        num = self.colors**self.positions   
        for i in range(num):
            rules.append(ruleType%self.colors)
            ruleType = ruleType//self.colors
        return rules 

    def arr2Num(self,arr):
        """
        docstring for arr2Num
        """
        res = 0 
        for i in arr:
            res = self.colors*res + i 
        return res
    def automata(self,ruleType=0,tag=None):
        """
        docstring for automata
        """
        self.colors = 3
        self.positions = 3
        num = self.colors**(self.colors**3)
        length = 1 + len(str(num))
        # rules = self.getRules(ruleType)
        # print(rules)
        rules = np.random.randint(0,self.colors,self.colors**3)
        ruleType = self.arr2Num(rules)
        n = 501
        m = n

        image = np.ones((m,n,3),np.uint8)
        types =[[0,0,0],
                [255,255,255],
                [128,128,128]]
        types =[[255,255,0],
                [0,255,255],
                [255,0,255]]
        # types =[[255,0,0],
        #         [0,0,255]]

        celluar = np.zeros(n,np.uint8)
        celluar[n//2] = 1
        # random initialized first row

        # celluar = np.random.randint(0,self.colors,n)
        # tag     = self.arr2Num(celluar)


        for i in range(n):
            image[0,i,:] = types[celluar[i]]

        coefs = np.array([self.colors**2,self.colors,1])
        for i in range(1,m):
            tmp = celluar.copy()
            # first column
            num = coefs[1]*tmp[0]+coefs[2]*tmp[1]
            num = rules[num]
            celluar[0] = num 
            image[i,0,:] = types[num]
            # middle columns 
            for j in range(1,n-1):
                num  = sum(coefs*tmp[j-1:j+2])
                num = rules[num]
                celluar[j] = num 
                image[i,j,:] = types[num]
            # last column
            num = coefs[0]*tmp[-2]+coefs[1]*tmp[-1]
            num = rules[num]
            celluar[-1] = num 
            image[i,-1,:] = types[num]

        image = Image.fromarray(image)
        name = str(ruleType)
        name = "0"*(length-len(name))+name
        if tag is None:
            imageName = "automata%s_%d.png"%(name,self.colors)
        else:
            imageName = "automata%s_%d.png"%(name,tag)

        print("save to ",imageName)
        image.save(imageName)


        return

    def testAutomata(self):
        """
        docstring for testAutomata
        """
        begin = 50
        end   = 256
        # for i in range(begin,end):
        sets = [57,73,86,150,165,
                126,52,129,99,109]

        np.random.seed(2)
        # for i in sets:
        #     self.automata(ruleType=i)

        for i in range(300):
            self.automata(ruleType=129)


        return
    def test(self):
        """
        docstring for test
        """
        self.testAutomata()
        return

class EggCurve():
    """
    get the analytic and numeric solutioin of 
    egg curve
    """
    def __init__(self):
        """
        self.a:
            the sum is 2*a
        self.c:
            two points (-c,0) and (c,0)
        self.beta:
            l2+beta*l1 = 2*a (egg curve, a ellepic one 
            when beta = 1)
        """
        super(EggCurve, self).__init__()
        self.a    = 2 
        self.c    = 1
        self.beta = 100
        self.count = 0
        return 

    def setParas(self,a,c,beta):
        """
        docstring for setParas
        """
        self.a    = a   
        self.c    = c   
        self.beta = beta

        return
    def analyticSol(self):
        """
        docstring for analyticSol
        """
        
        # formula:
        #    (1-beta**2)*l1**2+(4*a*beta - 4*c*cos(theta))*l1+
        #    4*c**2 - 4*a**2 = 0

        a    = sympy.Symbol("a")
        c    = sympy.Symbol("c")
        x    = sympy.Symbol("x")
        beta = sympy.Symbol("beta")
        theta = sympy.Symbol("theta")
        print(a)
        print(beta)
        equation  = (1-beta**2)*x**2
        equation += (4*a*beta - 4*c*sympy.cos(theta))*x
        equation += 4*c**2 - 4*a**2
        y = sympy.solve(equation,x)
        y = sympy.simplify(y)
        print(sympy.latex(y[0]))
    def numSol(self):
        """
        docstring for numSol
        """
        num = 500
        x = np.zeros(num*2)
        x[:num] = np.linspace(-1,0.9,500)
        x[num:] = np.linspace(0.9,1,500)
        y = []
        for t in x:
            y.append(self.getEggValue(t))
        y = np.array(y)
        plt.axis("equal")
        plt.plot(y[:,0],y[:,1])
        plt.plot(y[:,0],-y[:,1])
        return
    def showImage(self):
        """
        docstring for showImage
        """
        # plt.grid(True)
        name = "new%d.png"%(self.count)
        print(name)
        plt.savefig(name,dpi=400)
        plt.show()
        return
    def getEggValue(self,t):
        """
        docstring for getEggValue
        t = cos(theta)
        return:
            (x,y) <-- (l1*cos(theta) - c, l1*sin(theta))
        """
        # get l1
        A = self.beta**2 - 1 
        B = self.a*self.beta - self.c*t 
        C = self.a*self.a - self.c*self.c 
        delta = np.sqrt(B*B - A*C)
        L = []
        L.append(2*(B + delta)/A)
        L.append(2*(B - delta)/A)
        # if L[0] > 0 and L[0] < self.a:
        #     l1 = L[0]
        # else:
        #     l1 = L[1]
        l1 = min(L)
        if l1 < 0:
            l1 = sum(L) - l1
            

        # if t > 0.9:
        #     print(L)
        x = l1*t - self.c
        y = l1*np.sqrt(1-t*t)
        
        return [x,y]
    def run(self):
        """
        docstring for run
        """
        # for i in range(50):
        #     self.beta  = 0 + 0.02*i
        #     self.count = i+1
        #     self.numSol()
        for i in range(100):
            self.beta  = 1.0 + 0.1*(i+1)
            self.count = i+1
            self.numSol()
        self.showImage()
        return

class Mandelbrot():
    """
    docstring for Mandelbrot
    Z_n = Z_n^2 + C
    |Z_n| < \delta

    usage:
        mandel = Mandelbrot()
        mandel.run()
    """
    def __init__(self,parameters):
        super(Mandelbrot, self).__init__()
        self.setParas(parameters)
        
    def setParas(self,parameters):
        """
        docstring for setParas
        self.C:
            usually a complex number
        self.maxIter:
            maximum iteration number
        self.func_range:
            function range
        self.threshold:
            threshold of the system
        self.n:
            split number
        self.filename:
            output file name 
        self.Z:
            initial value, usually 0.0
        """
        
        self.C          = parameters["C"]  
        self.maxIter    = parameters["maxIter"]
        self.func_range = parameters["func_range"]
        self.threshold  = parameters["threshold"]
        self.n          = parameters["n"]
        self.filename   = parameters["filename"]
        self.Z = 0.0
        return   
    def get_iteration(self):
        # print(i)
        iteration = 0 
        while abs(self.Z)**2 < self.threshold and \
              iteration < self.maxIter:
            self.Z = self.Z**2 + self.C
            iteration += 1
        return iteration
    def get_mandelbrot(self):
        xValue = np.linspace(self.func_range[0],
                             self.func_range[1],
                             self.n)
        yValue = np.linspace(self.func_range[2],
                             self.func_range[3],
                             self.n)
        iterations = np.zeros((self.n,self.n))
        # get iterations
        print("calculating data")
        
        for i,x in tqdm(enumerate(xValue)):
            for j,y in enumerate(yValue):
                self.Z = x + y*1.0j
                iterations[i,j] = self.get_iteration()

        # get plot
        data = (xValue,yValue,iterations,self.maxIter)
        print("get plotting")
        
        # plot(self.filename,data)
    def run(self):
        """
        docstring for run
        run an instance
        """
        parameters = {}
        parameters["C"] = 2.0
        parameters["maxIter"] = 80
        parameters["threshold"] = 2.0
        parameters["func_range"] = [-5,5,-5,5]
        parameters["n"] = 256
        parameters["filename"] = "mandelbrot.txt"
        self.setParas(parameters)
        self.get_mandelbrot()
        return

class Manikon():

    """Docstring for Manikon. """

    def __init__(self):
        """TODO: to be defined. """
        super(Manikon,self).__init__()

    def checkPoint(self,res,a,b):
        """
        check the point res (x,y)
        (x,y), (a,b) --> 
        return x^2/a^2 + y^2/b^2
        """
        print("output")
        axis = np.array([a**2,b**2])
        output = res**2/axis
        output = np.sum(output,axis=1)
        print(output)
    
    def getTangentPoint(self,data,length,theta,a,b):
        """
        x0 = a*cos(theta)
        y0 = b*cos(theta)
        x*cos(theta)/a + y*sin(theta)/b = 1
        k1*x + k2*y = 1
        k1 = cos(theta) / a
        k2 = sin(theta) / b
    
        norm = sqrt(k1**2+k2**2)
        cos(alpha) = -k2/norm
        sin(alpha) =  k1/norm
        
        x1 = x0+length*cos(alpha)
        y1 = y0+length*sin(alpha)
        
        """
    
        # get tangent
        
        tangent = np.zeros((len(data),2))
    
        tangent[:,0] = np.cos(theta)/a
        tangent[:,1] = np.sin(theta)/b
        # get norm
        norm = np.sum(tangent**2,axis=1)
        norm = np.sqrt(norm)
        norm = np.repeat(norm,2)
        norm = np.reshape(norm,(-1,2))
        print(norm)
        tangent = tangent / norm
        # exchange the columns
        print("1",tangent)
        tmp = np.zeros(len(data))
        tmp[:] = tangent[:,0]
        tangent[:,0] = tangent[:,1]
        ###### error here!!!!!!!
        #tangent[:,1] = tmp
        ###### error here!!!!!!!
        tangent[:,1] = - tmp
        print("2",tangent)
        
        res = data - length*tangent
    
        print(res)
        self.checkPoint(res,a,b)
        print("data",data)
        self.checkPoint(data,a,b)
    
        return res
    
    def run(self):
        """TODO: Docstring for run.
        :returns: TODO

        """
    
        length = 100
        size = length
        res = np.zeros((length+1,2))
        theta = np.arange(length)*2*np.pi/length
        data = np.zeros((length,2))
        a = 4
        b = 2
        
        data[:,0] = a*np.cos(theta)
        data[:,1] = b*np.sin(theta)
        
        # tangent cos(alhpa) and sin(alpha)
        tangent = np.zeros((length,2))
        length = 3
        
        res[:size,:] = self.getTangentPoint(data,length,theta,a,b)
        
        res[-1,:] = res[0,:]
        
        filename = "manikon.txt"
        np.savetxt(filename,res,fmt="%.3f,%.3f")
        return 
        