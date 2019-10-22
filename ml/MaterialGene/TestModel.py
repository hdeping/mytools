#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-09-25 17:42:41
    @project      : material genome predition
    @version      : 1.0
    @source file  : TestModel.py

============================
"""

from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from globalTest import *
from .MaterialData import MaterialData
from .MaterialModel import MaterialModel
from args import args

class TestModel():

    """Docstring for TestModel. """

    def __init__(self):
        """
        TODO: to be defined1. 
        self.test_loader:
            data loader
        self.model:
            model of the neural networks
        """
        super(TestModel,self).__init__()

        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        
        cuda_args = {'num_workers': 1, 'pin_memory': True}
        kwargs =  cuda_args if args.cuda else {}
        test_data  = MaterialData(train=False)
        
        self.test_loader = data.DataLoader(test_data,
                                           batch_size=args.batch_size, 
                                           shuffle=False, 
                                           **kwargs)
        
        self.model = Network()
        if args.cuda:
            self.model.cuda()
        
        minimum = test_data.minimum
        maximum = test_data.maximum
    def test(number):
        """
        input:
            number, serial number for the output file
        return:
            None, but the result is written into a file
            named "test_result%d.txt"%(number)
        """
        #output = torch.Tensor
        model.eval()
        i = 0
        filename = "test_result%d.txt"%(number)
        fp = open(filename,'w')
        for (test_data,test_target) in self.test_loader:
            i = i + 1
            #if i > 1:
            #    break
            test_data = test_data.cuda()
            output = model(test_data)
            #fp.write(str(output))
            #print((output+1.0)*50)
            #print(i)
            for ii in range(len(output)):
                #print(output[ii].item(),test_target[ii].item())
                fp.write("%lf,%lf\n"%(output[ii].item(),test_target[ii].item()))
            #print(output*100)
            #print("length is",len(output))
            #break
        fp.close()
        return
    def run(self):
        """TODO: Docstring for run.
        :returns: TODO
        load parameters from .pt file
        and run the test method
        """

        for i in range(5,101,5):
            print(i)
            name = "models/modelnew"+str(i)+".pt"
            model.load_state_dict(torch.load(name))
            test(i)
