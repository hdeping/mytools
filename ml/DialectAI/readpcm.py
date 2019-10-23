# -*- coding:utf-8 -*-

"""a module to read the pcm format file"""

__author__ = 'hdeping'
__email__ = 'xiaohengdao@gmail.com'
__version__ = '2018.08.08 with python 3.6'

import numpy as np
import scipy 


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
def getPowerDensity(data):
    # windows function
    size = len(data[0])
    W_k = np.arange(size)
    W_k = 1+np.cos(2*np.pi*W_k/size)
    powerDensity= []
    for line in data:
        # fft
        line = scipy.fft(line)
        # abs
        line = np.abs(line)
        # power density spectrum
        line = line**2
        # windowed powerDensity
        line = line*W_k
        # normalized powerDensity
        #print(line.shape)
        summation = sum(line)
        if summation == 0:
            continue
        line = line/sum(line)
        print(summation)
        powerDensity.append(line)
    # return powerDensity
    return powerDensity
def getRt(powerDensity):
    R_t = []
    for line in powerDensity:
        # fft
        line = scipy.ifft(line)
        R_t.append(line)
    # get abs and
    # convert to numpy type
    R_t = np.abs(R_t)
    return R_t

def getTone(R_t,toneLengthD):
    # length of the column
    size =  len(R_t[0])
    # length of the row
    piece = len(R_t)
    tone = np.zeros((piece - 1,2*toneLengthD+1))

    const = 1e8
    # get tone

    for i in range(piece-1):
        # d < 0
        for j in range(toneLengthD):
            d = j - toneLengthD
            dim = abs(d)
            # get tone[i,j]
            arr1 = R_t[i,dim:]
            arr2 = R_t[i+1,:-dim]
            #print(len(arr1),len(arr2),size)
            scale = 1/(size - d)
            tone[i,j] = const*scale*(sum(arr1*arr2) - scale**2*sum(arr1)*sum(arr2))
        # d > 0
        for j in range(toneLengthD,2*toneLengthD+1):
            d = j - toneLengthD
            dim = abs(d)
            # get tone[i,j]
            arr1 = R_t[i,:(size-dim)]
            arr2 = R_t[i+1,dim:]
            scale = 1/(size - d)
            tone[i,j] = const*scale*(sum(arr1*arr2) - scale**2*sum(arr1)*sum(arr2))
        # d > 0
    # return tone
    return tone
            
    

def getToneFeature(data,toneLengthD):
    # get power density spectrum
    powerDensity = getPowerDensity(data)
    # get R_t(k)
    R_t = getRt(powerDensity)
    # get tone feature
    tone = getTone(R_t,toneLengthD)
    return tone



class pcmdata(object):

    # init function
    def __init__(self, path,windows=400,stride=160,toneLengthD=5):
        # read data
        data = np.memmap(path,dtype='h',mode='r')
        # get data and frame
        self.windows = windows
        self.stride  = stride
        self.data,self.frame = sliceArray(data,self.windows,self.stride)
        self.data = getToneFeature(self.data,toneLengthD)
        print(self.data.shape)
        self.frame = len(self.data)

    def read_data(self):
        return self.data

    # get frame number
    def get_frame_num(self):
        return self.frame
# -*- coding:utf-8 -*-
            
    

def getToneFeature(data,toneLengthD):
    # get power density spectrum
    powerDensity = getPowerDensity(data)
    # get R_t(k)
    R_t = getRt(powerDensity)
    # get tone feature
    tone = getTone(R_t,toneLengthD)
    return tone



class pcmdata(object):

    # init function
    def __init__(self, path,windows=400,stride=160,dimension=13):
        # read data
        data = np.memmap(path,dtype='h',mode='r')
        # get data and frame
        self.windows = windows
        self.stride  = stride
        self.data,self.frame = sliceArray(data,self.windows,self.stride)
        toneLengthD = dimension // 2
        # feature
        self.data = getToneFeature(self.data,toneLengthD)
        self.frame -= 1

    def read_data(self):
        return self.data

    # get frame number
    def get_frame_num(self):
        return self.frame
# -*- coding:utf-8 -*-
