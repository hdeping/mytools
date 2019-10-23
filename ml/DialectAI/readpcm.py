# -*- coding:utf-8 -*-

"""a module to read the pcm format file"""

__author__ = 'hdeping'
__email__ = 'xiaohengdao@gmail.com'
__version__ = '2018.08.08 with python 3.6'

import numpy as np



class pcmdata(object):

    # init function
    def __init__(self, path,dimension):
        data = np.memmap(path,dtype='h',mode='r')
        
        row = len(data) // dimension 
        self.data = np.reshape(data[:dimension*row],(row,dimension))
        self.row = row

    def read_data(self):
        return self.data

    # get frame number
    def get_frame_num(self):
        return self.row
