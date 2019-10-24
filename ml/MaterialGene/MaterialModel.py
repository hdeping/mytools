#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-09-25 18:18:00
                    2019-10-22 10:33:28
    @project      : Material Genome Project
    @version      : 1.0
    @source file  : MaterialModel.py

============================
"""

import torch.nn as nn
import torch.nn.functional as F

class MaterialModel(nn.Module):
    """
    model definition for the material genome project
    """
    def __init__(self):
        """
        self.layer[1-4]:
            conv3d_batch layers,1,16,32,64,128
        self.layer5:
            linear and batchnorm
        self.layer6:
            linear layer
        """
        super(Net, self).__init__()
        # layer1: convTran3d + batchnorm
        self.layer1 = self.conv3d_batch(1,16)
        self.layer2 = self.conv3d_batch(16,32)
        self.layer3 = self.conv3d_batch(32,64)
        self.layer4 = self.conv3d_batch(64,128)

        # (batch_size,128,7) -> (batch_size,128*7)
        node_num = 60
        self.layer5 = self.linear_batch(128*7,node_num)
        self.layer5.add_module('batchnorm', nn.BatchNorm1d(node_num))
        # layer5: linear + batchnorm
        # 30 -> 1 
        self.layer6 = self.linear_batch(node_num,1)

    def linear_batch(self,input_chanel,output_chanel):
        """
        input:
            input_channel, number of the input channel
            output_channel, number of the output channel
        return:
            linear_net, linear layer 
        """
        linear_net = nn.Sequential()
        linear_net.add_module('linear', nn.Linear(input_chanel,output_chanel))
        return linear_net
        
    def convTran3d_batch(self,input_chanel,output_chanel):
        """
        input:
            input_channel, number of the input channel
            output_channel, number of the output channel
        return:
            conv_net, ConvTranspose3d --> BatchNorm3d, 
            a two layer neural network
        """
        conv_net = nn.Sequential()
        model = nn.ConvTranspose3d(input_chanel,output_chanel,kernel_size=2)
        conv_net.add_module('conv',model) 
        model = nn.BatchNorm3d(output_chanel)
        conv_net.add_module('batchnorm', model)
        return conv_net

    def conv3d_batch(self,input_chanel,output_chanel):
        """
        input:
            input_channel, number of the input channel
            output_channel, number of the output channel
        return:
            conv_net, Conv3d --> BatchNorm3d, 
            a two layer neural network
        """
        conv_net = nn.Sequential()
        model = nn.Conv3d(input_chanel,output_chanel,kernel_size=3)
        conv_net.add_module('conv', model)
        conv_net.add_module('batchnorm', nn.BatchNorm3d(output_chanel))
        return conv_net

    def forward(self, x):
        """
        forward computation of the neural network
        input:
            x, 5D array, with the shape of (batch_size, d,m,n,s)
        return:
            x, result of the forward computation,
            2D array which shape (batch_size, 1)
        """
        # expand the input
        # size = 5
        #print(x.shape)
        x = self.expandMat(x,5)
        #print("expanded:",x.shape)
        # layer1
        x = self.layer1(x)
        x = F.relu(x)
        #print("layer1  :",x.shape)
        # layer2
        x = self.layer2(x)
        x = F.max_pool3d(x,2)
        x = F.relu(x)
        # layer3
        x = self.layer3(x)
        x = F.relu(x)
        # layer4
        x = self.layer4(x)
        x = x.squeeze()
        x = F.max_pool1d(x,2)
        x = F.relu(x)
        
        batch_size = x.size(0)

        # reshape the output tensor in order to be 
        # suitable as an input for a linear layer
        # layer5
        x = x.contiguous().view(batch_size,-1)
        x = self.layer5(x)
        x = F.relu(x)
        x = self.layer6(x)
        x = F.relu(x)
        return x
    def expandMat2(self,x,size):
        """
        input:
            x, 3D array, with the shape of (batch_size, m,n)
            size, an integer number
        return:
            x, 3D array, with the shape of 
            (batch_size, size*m,size*n)
        for instance:
            (x[2,3,4],10) --> x[20,30,40]
        """
        # rows and columns (m,n)
        batch_size , m,n = x.size()
        y = torch.zeros(batch_size,m*size,n*size)
        for i in range(size):
            for j in range(size):
                i_begin = i*m
                i_end   = (i+1)*m
                j_begin = j*n
                j_end   = (j+1)*n
                y[:,i_begin:i_end,j_begin:j_end] = x
        return y
    def expandMat(self,x,size):
        """
        input:
            x, 5D array, with the shape of (batch_size, d,m,n,s)
            size, an integer number
        return:
            x, 5D array, with the shape of 
            (batch_size, d,size*m,size*n,size*s)
        for instance:
            (x[2,3,4,5,6],10) --> x[20,30,40,50,60]
        """
        # rows and columns (m,n)
        batch_size ,d, m,n,s = x.size()
        y = torch.zeros(batch_size,d,m*size,n*size,s*size)
        y = y.cuda()
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    i_begin = i*m
                    i_end   = (i+1)*m
                    j_begin = j*n
                    j_end   = (j+1)*n
                    k_begin = k*s
                    k_end   = (k+1)*s
                    y[:,:,i_begin:i_end,j_begin:j_end,k_begin:k_end] = x
        return y
