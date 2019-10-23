# -*- coding:utf-8 -*-

import codecs
import copy
import random

import torch
import numpy as np

from readpcm import pcmdata,HTKfile


class TorchDataSet(object):
    def __init__(self, file_list, batch_size, chunk_num, dimension):
        self._batch_size = batch_size
        self._chunck_num = chunk_num
        self._chunck_size = self._chunck_num*self._batch_size
        self._dimension = dimension
        self._file_point = codecs.open(file_list, 'r', 'utf-8')
        self._dataset = self._file_point.readlines()
        self._file_point.close()
        random.shuffle(self._dataset)

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

            pcm_file = pcmdata(htk_feature)
            htk_file = HTKfile(htk_feature)
            feature_tone = pcm_file.read_data()
            feature_fb   = htk_file.read_data()
            # len1 < len2
            len1 = len(feature_tone)
            len2 = len(feature_fb)
            begin = len2 - len1
            feature_fb = feature_fb[begin:]
            #feature_data = np.concatenate((feature_fb,feature_tone),axis=1)
            #file_name = pcm_file.get_file_name()
            feature_frames = len1
            #print(feature_frames.shape)

            if feature_frames > max_frames:
                max_frames = feature_frames
            
            curr_feature_fb = torch.Tensor(feature_fb)
            curr_feature_tone = torch.Tensor(feature_tone)
            # normalization
            #curr_feature = curr_feature.mul(scale)
            means = curr_feature_fb.mean(dim=0, keepdim=True)
            curr_feature_fb = curr_feature_fb - means.expand_as(curr_feature_fb)
            means = curr_feature_tone.mean(dim=1, keepdim=True)
            curr_feature_tone = curr_feature_tone - means.expand_as(curr_feature_tone)


            # concate the two arrays
            curr_feature_norm = torch.cat((curr_feature_fb,curr_feature_tone),1)
            batch_data.append(curr_feature_norm)
            target_frames.append(torch.Tensor([target_label, feature_frames]))
            #name_list.append(file_name)

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

