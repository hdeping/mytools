# -*- coding:utf-8 -*-

"""a module to read the pcm format file"""

__author__ = 'hdeping'
__email__ = 'xiaohengdao@gmail.com'
__version__ = '2018.08.08 with python 3.6'

import numpy as np



class pcmdata(object):

    # init function
    def __init__(self, path,window_size=400):
        data = np.memmap(path,dtype='h',mode='r')
        
        row = len(data) // window_size
        self.data = np.reshape(data[:window_size*row],(row,window_size))
        self.row = row

    def read_data(self):
        return self.data

    # get frame number
    def get_frame_num(self):
        return self.row
