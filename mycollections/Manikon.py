#!/usr/bin/python

import numpy as np

class Manikon():

    """Docstring for Manikon. """

    def __init__(self):
        """TODO: to be defined. """
        super(Manikon,self).__init__()

    def checkPoint(self,res,a,b):
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
