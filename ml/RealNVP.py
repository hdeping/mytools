#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-12-27 11:19:19
    @project      : RealNVP
    @version      : 1.0
    @source file  : real-nvp-pytorch.py
    @source       :

============================
"""




import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 10, 8
rcParams['figure.dpi'] = 300

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler


# isCuda = torch.cuda.is_available()
isCuda = False


class RealNVP(nn.Module):
    """
    RealNVP: 
        real-valued non-volume preserving
        @article{Dinh2017,
            author = {Dinh, Laurent},
            eprint = {arXiv:1605.08803v3},
            journal = {ICLR},
            pages = {1--33},
            title = {{Density Estimation Using Real NVP}},
            year = {2017}
        }

    nets: function
    nett: inversed function
    mask:
    prior:

    """
    def __init__(self, mask, prior):
        super(RealNVP, self).__init__()
        
        self.prior = prior
        nets,nett = self.getModels()
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])

        if isCuda:
            self.t = self.t.cuda()
            self.s = self.s.cuda()
            mask   = mask.cuda()

        self.mask = nn.Parameter(mask, requires_grad=False)

        # self.loadTopytalModel()
        
    def g(self, z):
        """
        generator function
        """
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self,x):
        z, logp = self.f(x)
        if isCuda:
            tmp = self.prior.log_prob(z.cpu())
            return tmp.cuda() + logp
        else:
            return self.prior.log_prob(z) + logp
        
    def sample(self, batchSize): 
        """
        sample from the generator
        """
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        if isCuda:
            z = z.cuda()
        x = self.g(z)
        if isCuda:
            x = x.cpu()

        return x 

    def getModels(self):
        
        nets = lambda: nn.Sequential(nn.Linear(2, 256), 
                                     nn.LeakyReLU(), 
                                     nn.Linear(256, 256), 
                                     nn.LeakyReLU(), 
                                     nn.Linear(256, 2), 
                                     nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(2, 256), 
                                     nn.LeakyReLU(), 
                                     nn.Linear(256, 256), 
                                     nn.LeakyReLU(), 
                                     nn.Linear(256, 2))

        # nets = self.loadModel(nets,"s0.pt")
        # nett = self.loadModel(nets,"t0.pt")

        return nets,nett

    def loadModel(self,net,filename):
        """
        load the parameters from a file
        """  
        
        fp = torch.load(filename)
        model = net()
        model.load_state_dict(fp)
        return model

    def loadTotalModel(self):
        print("load parameters from .pt file")
        
        fp = torch.load("s0.pt")
        self.s.load_state_dict(fp)
        fp = torch.load("t0.pt")
        self.t.load_state_dict(fp)
        return

    def saveModelPara(self):
        """
        docstring for saveModelPara
        save the parameters of the models
        """
        # for i in range(len(self.mask)):
        print("save parameters into files")
        for i in range(1):
            
            torch.save(self.s.state_dict(),"s%d.pt"%(i))
            torch.save(self.t.state_dict(),"t%d.pt"%(i))
        return

    def getData(self):
        data = datasets.make_moons(n_samples=100, noise=.05)[0]
        data = data.astype(np.float32)
        data = torch.from_numpy(data)
        if isCuda:
            data = data.cuda()
        return data 

    def plot(self):
        noisy_moons = self.getData()
        z = flow.f(noisy_moons)[0]
        if isCuda:
            z = z.cpu().detach().numpy()
        else:
            z = z.detach().numpy()
        plt.subplot(221)
        plt.scatter(z[:, 0], z[:, 1])
        plt.title(r'$z = f(X)$')

        z = np.random.multivariate_normal(np.zeros(2), np.eye(2), 1000)
        plt.subplot(222)
        plt.scatter(z[:, 0], z[:, 1])
        plt.title(r'$z \sim p(z)$')

        plt.subplot(223)
        x = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
        plt.scatter(x[:, 0], x[:, 1], c='r')
        plt.title(r'$X \sim p(X)$')

        plt.subplot(224)
        x = self.sample(1000).detach().numpy()
        plt.scatter(x[:, 0, 0], x[:, 0, 1], c='r')
        plt.title(r'$X = g(z)$')

        plt.savefig("output.png",dvi=300)
        plt.show()

    def train(self):
        """
        train the model
        """
        theta = [p for p in self.parameters() if p.requires_grad==True]
        optimizer = torch.optim.Adam(theta, lr=1e-4)
        for t in range(5001):    
            noisy_moons = self.getData()
            loss = -self.log_prob(noisy_moons).mean()
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            if t % 200 == 0:
                print('iter %s:' % t, 'loss = %.3f' % loss)


masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 4).astype(np.float32))
prior = distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))


flow = RealNVP(masks, prior)

flow.train()

flow.plot()

flow.saveModelPara()
