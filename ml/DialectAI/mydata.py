# -*- coding:utf-8 -*-

import copy
import random
import numpy as np

import torch

from readhtk import HTKfile


class TorchDataSet(object):
    def __init__(self, file_list, batch_size, chunk_num, dimension,p_vad):
        self._batch_size = batch_size
        self._chunck_num = chunk_num
        self._chunck_size = self._chunck_num*self._batch_size
        self._dimension = dimension
        self._filenames = np.loadtxt(file_list,delimiter=' ',dtype=str)
        self._p_vad = p_vad
        random.shuffle(self._filenames)

    def reset(self):
        random.shuffle(self._filenames)
    
    def __iter__(self):
        batch_data = []
        target_frames = []
        name_list = []
        max_frames = 0
        for ii,names in enumerate(self._filenames):
            htk_feature = names[:4]
            #print("ii = ",ii)
            target_label = int(str(names[-1])) 

            htk_file = HTKfile(htk_feature,self._p_vad)
            feature_data = htk_file.read_data()
            file_name = htk_file.get_file_name()
            feature_frames = htk_file.get_frame_num()

            if feature_frames > max_frames:
                max_frames = feature_frames
            
            curr_feature = torch.Tensor(feature_data)
            means = curr_feature.mean(dim=0, keepdim=True)
            curr_feature_norm = curr_feature - means.expand_as(curr_feature)
            batch_data.append(curr_feature_norm)
            target_frames.append(torch.Tensor([target_label, feature_frames]))
            name_list.append(file_name)

            if (ii+1) % self._chunck_size == 0:
                chunk_size = len(batch_data)
                idx = 0
                data = torch.zeros(self._batch_size, max_frames, self._dimension)
                target = torch.zeros(self._batch_size, 2)
                for jj in range(chunk_size):
                    curr_data = batch_data[jj]
                    curr_tgt = target_frames[jj]
                    curr_frame = curr_data.size(0)

                    data[idx,:curr_frame,:] = curr_data[:,:]
                    target[idx,:] = curr_tgt[:]
                    idx += 1

                    if idx % self._batch_size == 0:
                        idx = 0
                        yield data, target
                
                max_frames = 0
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
                curr_data = batch_data[jj]
                curr_tgt = target_frames[jj]
                curr_frame = curr_data.size(0)

                data[idx,:curr_frame,:] = curr_data[:,:]
                target[idx,:] = curr_tgt[:]
                idx += 1

                if idx % self._batch_size == 0:
                    idx = 0
                    yield data, target

