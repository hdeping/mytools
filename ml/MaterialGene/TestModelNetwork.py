#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-22 10:13:39
    @project      : material genome project
    @version      : 1.0
    @source file  : TestModelNetwork.py

============================
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from .MaterialModel import MaterialModel


class TestModelNetwork():

    """Docstring for TestModel. """

    def __init__(self):
        """TODO: to be defined1. """
        super(TestModel,self).__init__()

        return

    def conv3d_batch(input_channel,output_channel):
        """
        input:
            input_channel, number of the input channel
            output_channel, number of the output channel
        return:
            conv_net, Conv3d --> BatchNorm3d, 
            a two layer neural network
        """
        conv_net = nn.Sequential()
        model = nn.Conv3d(input_channel,output_channel,kernel_size=3)
        conv_net.add_module('conv', model)
        model = nn.BatchNorm3d(output_channel)
        conv_net.add_module('batchnorm', model)
        return conv_net

    def getNewX(x):
        """
        input:
            x, torch array type
        return: 
            x, output of the neural networks
        """
        m = 3
        n = 3
        l = 8
        layer1 = conv3d_batch(1,16)
        layer2 = conv3d_batch(16,32)
        layer3 = conv3d_batch(32,64)
        layer4 = conv3d_batch(64,128)
        size = 5
        batch_size = 64
        x = torch.rand(batch_size,1,m*size,n*size,l*size)
        print("data   :",x.shape)
        x = layer1(x)
        x = layer2(x)
        x = F.max_pool3d(x,2)
        x = layer3(x)
        x = layer4(x)
        x = x.squeeze()
        x = F.max_pool1d(x,2)
        x = x.view(batch_size,-1)
        #print("layer4 :",x.shape)
        #print("layer  :",x.shape)

        return x
    def test(self):
        """TODO: Docstring for test.
        :returns: TODO
        test the MaterialModel
        """

        size = 1
        x = torch.rand(batch_size,1,m*size,n*size,l*size)
        model = MaterialModel()
        y = model(x)
        return