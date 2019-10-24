#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-09-25 18:18:00
    @project      : Material Genome Project
    @version      : 1.0
    @source file  : MaterialData.py

============================
"""

import torch.utils.data as data
import torch
import numpy as np 
from .Wyckoff import Wyckoff


class MaterialData(data.Dataset,Wyckoff):
    def __init__(self,train=True):
        super(MaterialData,self).__init__()
        self.train = train  # training set or test set
        self.trainNum = 160000
        self.getInputData()
        self.loadData()
        
    # get the input_data and labels
    def loadData():
        if self.train:
            self.train_data   = input_data[0:self.trainNum]
            self.train_labels = labels[0:self.trainNum]
        else:
            self.test_data   =  input_data[self.trainNum:]
            self.test_labels =  labels[self.trainNum:]

        return 
    def __getitem__(self,index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img,target
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def getInputData(self):
        # the input file 
        # read the data into a array
        # the first 11 columns are the input
        # the last one is the labels
        # get data
        data,labels = self.getFullMatrix()
        length = len(data)
        input_data = np.reshape(data,(length,1,3,3,8))
        # transfer the data from numpy to torch
        self.input_data = torch.from_numpy(input_data)
        # normalization 
        self.minimum = min(labels)
        self.maximum = max(labels)
        # map the energies into the range [0,1]
        labels = (labels - minimum)/(maximum - minimum)
    
        labels = torch.from_numpy(labels)
        self.labels = labels.type(torch.FloatTensor)
        return 
    
