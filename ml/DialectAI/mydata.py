# -*- coding:utf-8 -*-

import codecs
import copy
import random

import torch

from readhtk import HTKfile


class TorchDataSet(object):
    def __init__(self, file_list, batch_size, chunk_num, dimension,weight=True):
        self._batch_size = batch_size
        self._chunck_num = chunk_num
        self._chunck_size = self._chunck_num*self._batch_size
        self._dimension = dimension
        self._file_point = codecs.open(file_list, 'r', 'utf-8')
        self._weight = weight
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
            target_label  = int(str(splited_line[1])) 
            if self._weight:
                target_weight = int(str(splited_line[2])) 

            htk_file = HTKfile(htk_feature)
            feature_data = htk_file.read_data()
            file_name = htk_file.get_file_name()
            feature_frames = htk_file.get_frame_num()

            if feature_frames > max_frames:
                max_frames = feature_frames
            
            curr_feature = torch.Tensor(feature_data)
            means = curr_feature.mean(dim=0, keepdim=True)
            std   = curr_feature.std(dim=0, keepdim=True)
            # means
            curr_feature_norm = curr_feature - means.expand_as(curr_feature)
            # std
            curr_feature_norm = curr_feature_norm / std.expand_as(curr_feature)
            batch_data.append(curr_feature_norm)
            if self._weight:
                target_frames.append(torch.Tensor([target_label,target_weight, feature_frames]))
            else:
                target_frames.append(torch.Tensor([target_label,feature_frames]))
            name_list.append(file_name)

            if (ii+1) % self._chunck_size == 0:
                chunk_size = len(batch_data)
                idx = 0
                data = torch.zeros(self._batch_size, max_frames, self._dimension)
                if self._weight:
                    target = torch.zeros(self._batch_size, 3)
                else:
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
            if self._weight:
                target = torch.zeros(self._batch_size, 3)
            else:
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

