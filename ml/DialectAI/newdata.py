# -*- coding:utf-8 -*-

import copy
import random
import numpy as np
import codecs

import torch

from readhtk import HTKfile


class TorchDataSet(object):
    def __init__(self, file_list, batch_size, chunk_num, dimension,p_vad):
        self._batch_size = batch_size
        self._chunck_num = chunk_num
        self._chunck_size = self._chunck_num*self._batch_size
        self._dimension = dimension
        self._filenames = np.loadtxt(file_list,delimiter=' ',dtype=str)
        self._file_point = codecs.open(file_list, 'r', 'utf-8')
        self._dataset = self._file_point.readlines()
        self._file_point.close()
        self._p_vad = p_vad
        #random.shuffle(self._filenames)
        random.shuffle(self._dataset)

    def reset(self):
        #random.shuffle(self._filenames)
        random.shuffle(self._dataset)
    
    def __iter__(self):
        batch_data = []
        target_frames = []
        name_list = []
        max_frames = 20
        #for ii,names in enumerate(self._filenames):
        data_size = len(self._dataset)
        for ii in range(data_size):
            line = self._dataset[ii].strip() # what is this ? strip?
            names = line.split() # what is splitted?
            htk_feature = names[:4]
            print(names[0],names[-1])
            target_label = int(str(names[-1])) 


            batch_data.append(1)
            target_frames.append(20)
            name_list.append(names[0])

            if (ii+1) % self._chunck_size == 0:
                chunk_size = len(batch_data)
                idx = 0
                data = torch.zeros(self._batch_size, max_frames, self._dimension)
                target = torch.zeros(self._batch_size, 2)
                for jj in range(chunk_size):
                    idx += 1

                    if idx % self._batch_size == 0:
                        idx = 0
                        yield data, target
                
                batch_data = []
                target_frames = []
                name_list = []
            
            else:
                pass
            

        chunk_size = len(batch_data)
        if chunk_size > self._batch_size: 
            idx = 0
            data = torch.zeros(self._batch_size, max_frames, self._dimension)
            target = torch.zeros(self._batch_size, 2)
            for jj in range(chunk_size):
                idx += 1

                if idx % self._batch_size == 0:
                    idx = 0
                    yield data, target

