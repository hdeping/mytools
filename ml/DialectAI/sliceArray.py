# -*- coding:utf-8 -*-

"""a module to read the pcm format file"""

__author__ = 'hdeping'
__email__ = 'xiaohengdao@gmail.com'
__version__ = '2018.08.08 with python 3.6'

import numpy as np


def sliceArray(array,windows,stride):
    length_input = len(array)
    # (window,offset) = (10,2)
    size = (length_input - windows+stride)//stride
    # reshape a into c
    # if length_input is not divides by stride perfectly
    row = length_input // stride + 1
    tmp = np.zeros(row*stride)
    tmp[:len(array)] = array
    tmp = np.reshape(tmp,(row,stride))
    # get res
    # slice piece
    slice_num = windows // stride
    # residual piece
    res_num = windows - slice_num*stride
    res = np.zeros((size,windows))
    # the slicing part
    for i in range(slice_num):
        start_column = stride*i
        end_column = stride*(i+1)
        res[:,start_column:end_column] = tmp[i:i+size,:]
    # the residual part
    if res_num != 0:
        i = slice_num
        start_column = stride*slice_num
        end_column = windows
        res[:,start_column:end_column] = tmp[i:i+size,0:res_num]
    return res,size

arr = np.arange(1021)
res,size = sliceArray(arr,40,20)
print(res.shape,size)
print(res)
