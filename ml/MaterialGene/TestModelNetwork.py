#!/usr/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F
from .MaterialModel import MaterialModel


class TestModelNetwork():

    """Docstring for TestModel. """

    def __init__(self):
        """TODO: to be defined1. """
        super(TestModel,self).__init__()

    def conv3d_batch(input_chanel,output_chanel):
        conv_net = nn.Sequential()
        conv_net.add_module('conv', 
                nn.Conv3d(input_chanel,output_chanel,kernel_size=3))
        conv_net.add_module('batchnorm', 
                nn.BatchNorm3d(output_chanel))
        return conv_net

        layer1 = conv3d_batch(1,16)
        layer2 = conv3d_batch(16,32)
        layer3 = conv3d_batch(32,64)
        layer4 = conv3d_batch(64,128)

    def getNewX(x):
        m = 3
        n = 3
        l = 8
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
    def test(self):
        """TODO: Docstring for test.
        :returns: TODO

        """

        size = 1
        x = torch.rand(batch_size,1,m*size,n*size,l*size)
        model = MaterialModel()
        y = model(x)
