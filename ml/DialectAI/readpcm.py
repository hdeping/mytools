# -*- coding:utf-8 -*-

"""a module to read the pcm format file"""

__author__ = 'hdeping'
__email__ = 'xiaohengdao@gmail.com'
__version__ = '2018.08.08 with python 3.6'

import numpy as np



class pcmdata(object):

    # init function
    def __init__(self, path):
        data = np.memmap(path,dtype='h',mode='r')
        
        row = len(data) // 400
        self.data = np.reshape(data[:400*row],(row,400))
        self.row = row

    def read_data(self):
        return self.data

    # get frame number
    def get_frame_num(self):
        return self.row
