#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-09-06 19:12:55
    @project      : get some distance measures
    @version      : 0.1
    @source file  : DistanceMeasure.py

============================
"""
import numpy as np



class DistanceMeasure(object):
    """docstring for DistanceMeasure
    different kinds of distance measures
    input: number array p
           number array q
    return: a float number 
    """
    def __init__(self):
        super(DistanceMeasure, self).__init__()

    def Euclidean(self,p,q):
        d = np.linalg.norm(p-q)
        return d 

    def CityBlock(self,p,q):
        d = np.abs(p-q)
        d = sum(d)
        return d       
    def Minkowski(self,p,q,exponent):
        d = np.abs(p-q)
        d = sum(d**exponent)
        d = d**(1/exponent)
        return d

    def Chebyshev(self,p,q):
        d = np.abs(p-q)
        d = max(d)
        return d

    def Sorensen(self,p,q):
        d = sum(p+q)
        d = self.CityBlock(p,q)/d
        return d
  
    def Gower(self,p,q):
        d = self.CityBlock(p,q)/len(q)
        return d
  
    def Soergel(self,p,q):
        d = self.MaxDist(p, q)
        d = self.CityBlock(p,q)/d 
        return d

    
    def concate(self,p,q):
        """
        concatenate two arrays
        """
        p = np.reshape(p,(1,-1))
        q = np.reshape(q,(1,-1))
        res = np.concatenate((p,q))
        return res
    def KulczynskiD(self,p,q):
        d = self.MinDist(p, q)
        # print("d = ",d)
        
        d = self.CityBlock(p, q)/d
        return d

    def MaxDist(self,p,q):
        """
        max distance
        """

        res = self.concate(p,q)
        d = np.max(res,axis=0)
        d = sum(d)
        return d

    def MinDist(self,p,q):
        """
        min distance
        """
        res = self.concate(p,q)
        d = np.min(res,axis=0)
        d = sum(d)
        return d

    def Canberra(self,p,q):
        d = np.abs((p-q)/(p+q))
        d = sum(d)
        return d 

    def Lorentzian(self,p,q):
        d = np.log(1+np.abs(p - q))
        d = sum(d)
        return d 

    def Intersection(self,p,q):
        return self.MinDist(p, q)

    def WaveHedges(self,p,q):
        res = self.concate(p, q)
        res = np.max(res,axis=0)
        d   = np.abs(p - q) / res 
        d   = sum(d)
        return d 

    def Czekanowski(self,p,q):
        d = 2*self.MinDist(p,q)
        d = d / sum(p+q)
        return d 
    def Motyka(self,p,q):
        d = 0.5*self.Czekanowski(p,q)
        return d 

    def KulczynskiS(self,p,q):
        d = 1/self.KulczynskiD(p, q)
        return d 

    def Ruzicka(self,p,q):
        d = self.MinDist(p, q)
        d = d / self.MaxDist(p,q)
        return d 

    def Tanimoto(self,p,q):
        d = self.MaxDist(p, q)
        d = (d - self.MinDist(p, q))/d 

        return d

    def InnerProduct(self,p,q):
        d = np.dot(p,q)
        return d

    def HarmonicMean(self,p,q):
        d = p*q/(p+q)
        d = 2*sum(d)
        return d

    def Cosine(self,p,q):
        d = self.InnerProduct(p,q)      
        d = d / np.linalg.norm(p)
        d = d / np.linalg.norm(q)

        return d

    def Jaccard(self,p,q):
        d = self.InnerProduct(p,q)
        d = d / (sum(p**2+q**2) - d)
        return d

    def Dice(self,p,q):
        d = self.InnerProduct(p,q)
        d = 2*d/sum(p**2+q**2)

        return d
    def Fidelity(self,p,q):
        d = np.sqrt(p*q)
        d = sum(d)
        return d

    def Bhattacharyya(self,p,q):
        d = - np.log(self.Fidelity(p,q))
        return d

    def Hellinger(self,p,q):
        d = 2*np.sqrt(1 - self.Fidelity(p,q))
        return d

    def Matusita(self,p,q):
        d = self.Hellinger(p,q)/np.log(2)
        return d

    def SquaredChord(self,p,q):
        d = sum(p+q) - 2*self.Fidelity(p,q)
        return d

    def SquaredEuclidean(self,p,q):
        d = sum((p-q)**2)
        return d

    def Pearson(self,p,q):
        d = (p-q)**2 / p 
        d = sum(d)
        return d

    def Neyman(self,p,q):
        d = (p-q)**2 / q
        d = sum(d)
        return d 

    def Squared(self,p,q):
        d = (p-q)**2/(p+q)
        d = sum(d)
        return d

    def ProbSymmetric(self,p,q):
        d = 2*self.Squared(p,q)
        return d  

    def Divergence(self,p,q):
        d = (p-q)**2/(p+q)**2
        d = 2*sum(d)
        return d

    def Clark(self,p,q):
        d = self.Divergence(p,q)
        d = d/np.log(2)
        return d

    def AddSymmetric(self,p,q):
        d = (p-q)**2*(p+q)/(p*q)
        d = sum(d)
        return d 

    def KullbackLeibler(self,p,q):
        d = p*np.log(p/q)
        d = sum(d)
        return d

    def Jeffreys(self,p,q):
        d = (p - q)*np.log(p/q)
        d = sum(d)
        return d

    def KDivergence(self,p,q):
        d = p*np.log(2*p/(p+q))
        d = sum(d)
        return d  

    def Topsoe(self,p,q):
        d = p*np.log(2*p/(p+q))
        d = d + q*np.log(2*q/(p+q))
        d = sum(d)
        return d

    def JensenShannon(self,p,q):
        d  = self.KDivergence(p,q)
        d += self.KDivergence(q,p)
        d = d/2
        return d

    def Jensen(self,p,q):
        d = self.Entropy(p) + self.Entropy(q) 
        d = d - self.Entropy((p+q)/2)
        d = sum(d)
        return d 

    def Entropy(self,p):
        return p*np.log(p)


    def test(self):
        """
        test all the distance measures
        """
        # res = distance.concate(p,q)
        # print(res)
        # print(np.max(res,axis=0))
        # print(distance.Soergel(p,q))
        p = [0.1,0.2,0.3,0.4]
        q = [0.05,0.25,0.15,0.55]
        p = np.array(p)
        q = np.array(q)
        print("CityBlock: ")
        print(self.CityBlock(p,q))
        print("Soergel: ")
        print(self.Soergel(p,q))
        print("Gower: ")
        print(self.Gower(p,q))
        print("Chebyshev: ")
        print(self.Chebyshev(p,q))
        print("Minkowski(p,q: ")
        print(self.Minkowski(p,q,100))
        print("KulczynskiD: ")
        print(self.KulczynskiD(p,q))
        print("Euclidean: ")
        print(self.Euclidean(p,q))
        print("MinDist: ")
        print(self.MinDist(p,q))
        print("MaxDist: ")
        print(self.MaxDist(p,q))
        print("Canberra: ")
        print(self.Canberra(p,q))
        print("Lorentzian: ")
        print(self.Lorentzian(p,q))
        print("Intersection: ")
        print(self.Intersection(p,q))
        print("WaveHedges: ")
        print(self.WaveHedges(p,q))
        print("Czekanowski: ")
        print(self.Czekanowski(p,q))
        print("Motyka: ")
        print(self.Motyka(p,q))
        print("KulczynskiS: ")
        print(self.KulczynskiS(p,q))
        print("Ruzicka: ")
        print(self.Ruzicka(p,q))
        print("Tanimoto: ")
        print(self.Tanimoto(p,q))
        print("Jaccard: ")
        print(self.Jaccard(p,q))
        print("Dice: ")
        print(self.Dice(p,q))
        print("InnerProduct: ")
        print(self.InnerProduct(p,q))
        print("HarmonicMean: ")
        print(self.HarmonicMean(p,q))
        print("Cosine: ")
        print(self.Cosine(p,q))
        print("Fidelity: ")
        print(self.Fidelity(p,q))
        print("Bhattacharyya: ")
        print(self.Bhattacharyya(p,q))
        print("Hellinger: ")
        print(self.Hellinger(p,q))
        print("Matusita: ")
        print(self.Matusita(p,q))
        print("SquaredChord: ")
        print(self.SquaredChord(p,q))
        print("SquaredEuclidean: ")
        print(self.SquaredEuclidean(p,q))
        print("Pearson: ")
        print(self.Pearson(p,q))
        print("Neyman: ")
        print(self.Neyman(p,q))
        print("Squared: ")
        print(self.Squared(p,q))
        print("ProbSymmetric: ")
        print(self.ProbSymmetric(p,q))
        print("Divergence: ")
        print(self.Divergence(p,q))
        print("Clark: ")
        print(self.Clark(p,q))
        print("AddSymmetric: ")
        print(self.AddSymmetric(p,q))
        print("KullbackLeibler: ")
        print(self.KullbackLeibler(p,q))
        print("Jeffreys: ")
        print(self.Jeffreys(p,q))
        print("KDivergence: ")
        print(self.KDivergence(p,q))
        print("Topsoe: ")
        print(self.Topsoe(p,q))
        print("JensenShannon: ")
        print(self.JensenShannon(p,q))
        print("Jensen: ")
        print(self.Jensen(p,q))