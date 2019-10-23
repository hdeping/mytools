# -*- coding:utf-8 -*-

import codecs
import copy
import random

import torch

from readhtk import HTKfile
import numpy as np


class TorchDataSet(object):
    def __init__(self, file_list, batch_size, chunk_num, dimension):
        self._batch_size = batch_size
        self._chunck_num = chunk_num
        self._chunck_size = self._chunck_num*self._batch_size
        self._dimension = dimension
        self._file_point = codecs.open(file_list, 'r', 'utf-8')
        self._dataset = self._file_point.readlines()
        self._file_point.close()

    def reset(self):
        random.shuffle(self._dataset)
    
    def __iter__(self):
        data_size = len(self._dataset)
        batch_data = []
        target_frames = []
        name_list = []
        max_frames = 0
        for ii in range(data_size):
            line = self._dataset[ii].strip() # what is this ? strip?
            splited_line = line.split() # what is splitted?
            #print(splited_line)
            htk_feature = splited_line[0]
            #print("ii = ",ii)
            target_label = int(str(splited_line[1])) 

            # fb 40
            htk_file = HTKfile(htk_feature)
            feature_fb = htk_file.read_data()
            #print(feature_data.shape)
            file_name = htk_file.get_file_name()
            feature_frames = htk_file.get_frame_num()
            # plp0 13
            htk_file = HTKfile(htk_feature.replace('fb40','plp0'))
            feature_plp = htk_file.read_data()

            if feature_frames > max_frames:
                max_frames = feature_frames
            
            # concatenate fb40 and plp0
            feature_data = np.concatenate((feature_fb,feature_plp),axis=1)

            
            curr_feature = torch.Tensor(feature_data)
            means = curr_feature.mean(dim=0, keepdim=True)
            #std = curr_feature.std(dim=0, keepdim=True)
            # mean
            curr_feature_norm = curr_feature - means.expand_as(curr_feature)
            # std
            #curr_feature_norm = curr_feature_norm / std.expand_as(curr_feature)
            batch_data.append(curr_feature_norm)
            target_frames.append(torch.Tensor([target_label, feature_frames]))
            name_list.append(file_name)

            if (ii+1) % self._chunck_size == 0:
                chunk_size = len(batch_data)
                idx = 0
                data = torch.zeros(self._batch_size, max_frames, self._dimension)
                target = torch.zeros(self._batch_size, 2)
                names = np.array(['0000000000000000000000000000000000000000000000000000000000000000000000'])
                names = np.repeat(names,self._batch_size)
                for jj in range(chunk_size):
                    curr_data = batch_data[jj]
                    curr_tgt = target_frames[jj]
                    curr_frame = curr_data.size(0)

                    data[idx,:curr_frame,:] = curr_data[:,:]
                    target[idx,:] = curr_tgt[:]
                    names[idx] = name_list[jj]

                    idx += 1

                    if idx % self._batch_size == 0:
                        idx = 0
                        yield data, target,names
                
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
            names = np.array(['0000000000000000000000000000000000000000000000000000000000000000000000'])
            names = np.repeat(names,self._batch_size)
            for jj in range(chunk_size):
                curr_data = batch_data[jj]
                curr_tgt = target_frames[jj]
                curr_frame = curr_data.size(0)

                data[idx,:curr_frame,:] = curr_data[:,:]
                target[idx,:] = curr_tgt[:]
                names[idx] = name_list[jj]

                idx += 1

                if idx % self._batch_size == 0:
                    idx = 0
                    yield data, target,names

