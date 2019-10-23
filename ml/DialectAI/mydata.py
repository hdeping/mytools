# -*- coding:utf-8 -*-

import codecs
import copy
import random

import torch

from HTKfile import HTKfile
from getPhonemes2 import dealMlf


class TorchDataSet(object):
    def __init__(self, file_list, batch_size, chunk_num, dimension, dimension2, mlf_file):
        self._batch_size   = batch_size
        self._chunck_num   = chunk_num
        self._chunck_size  = self._chunck_num*self._batch_size
        self._dimension    = dimension
        self._dimension2   = dimension2
        self._file_point   = codecs.open(file_list, 'r', 'utf-8')
        self._dataset      = self._file_point.readlines()
        self._mlf_seq      = dealMlf(mlf_file)

        self._file_point.close()

    def reset(self):
        random.shuffle(self._dataset)
    
    def __iter__(self):
        data_size = len(self._dataset)
        batch_data = []
        target_frames = []
        name_list = []
        #max_frames_fb      = 0
        #max_frames_phoneme = 0
        # fb , phoneme
        max_frames = [0,0]

        for ii in range(data_size):
            line = self._dataset[ii].strip() # what is this ? strip?
            splited_line = line.split() # what is splitted?
            #print(splited_line)
            htk_feature = splited_line[0]
            #print("ii = ",ii)
            target_label = int(str(splited_line[1])) 

            # fb feature
            htk_file = HTKfile(htk_feature)
            feature_data_fb = htk_file.read_data()
            feature_frames_fb = htk_file.get_frame_num()
            # phoneme feature
            # get the sample name
            htk_feature = htk_feature.split('/')
            # last column
            htk_feature = htk_feature[-1]
            feature_data_phoneme = self._mlf_seq[htk_feature]
            file_name = htk_feature
            feature_frames_phoneme = len(feature_data_phoneme)

            # get max frames of the current chunk
            if feature_frames_fb > max_frames[0]:
                max_frames[0] = feature_frames_fb
            if feature_frames_phoneme > max_frames[1]:
                max_frames[1] = feature_frames_phoneme
            
            curr_feature_fb = torch.Tensor(feature_data_fb)
            curr_feature_phoneme = torch.Tensor(feature_data_phoneme)
            means = curr_feature_fb.mean(dim=0, keepdim=True)
            curr_feature_fb_norm = curr_feature_fb - means.expand_as(curr_feature_fb)
            # combine two vectors
            curr_feature = [curr_feature_fb_norm,curr_feature_phoneme]
            batch_data.append(curr_feature)
            # feature frames of fb and phoneme
            target_frames.append(torch.Tensor([target_label, feature_frames_fb,feature_frames_phoneme]))
            #print(target_label, feature_frames_fb,feature_frames_phoneme)
            name_list.append(file_name)

            if (ii+1) % self._chunck_size == 0:
                chunk_size = len(batch_data)
                idx = 0
                # fb
                data_fb = torch.zeros(self._batch_size, max_frames[0], self._dimension2)
                # phoneme
                data_phoneme = torch.zeros(self._batch_size, max_frames[1], self._dimension)
                # size of three: target , frame_fb and frame_phoneme
                target = torch.zeros(self._batch_size, 3)
                for jj in range(chunk_size):
                    curr_data = batch_data[jj]
                    curr_tgt = target_frames[jj]
                    # fb
                    curr_frame = curr_data[0].size(0)
                    #print(curr_frame)
                    data_fb[idx,:curr_frame,:] = curr_data[0][:,:]
                    # phoneme
                    curr_frame = curr_data[1].size(0)
                    #print(curr_frame)
                    data_phoneme[idx,:curr_frame,:] = curr_data[1][:,:]

                    target[idx,:] = curr_tgt[:]
                    #print(target[idx])
                    idx += 1

                    if idx % self._batch_size == 0:
                        idx = 0
                        yield data_fb, data_phoneme,  target
                
                max_frames = [0,0]
                batch_data = []
                target_frames = []
                name_list = []
            
            else:
                pass
            

        chunk_size = len(batch_data)
        if chunk_size > self._batch_size: 
            idx = 0
            # fb
            data_fb = torch.zeros(self._batch_size, max_frames[0], self._dimension2)
            # phoneme
            data_phoneme = torch.zeros(self._batch_size, max_frames[1], self._dimension)

            target = torch.zeros(self._batch_size, 3)
            for jj in range(chunk_size):
                curr_data = batch_data[jj]
                curr_tgt = target_frames[jj]

                # fb
                curr_frame = curr_data[0].size(0)
                data_fb[idx,:curr_frame,:] = curr_data[0][:,:]
                # phoneme
                curr_frame = curr_data[1].size(0)
                data_phoneme[idx,:curr_frame,:] = curr_data[1][:,:]

                target[idx,:] = curr_tgt[:]

                idx += 1

                if idx % self._batch_size == 0:
                    idx = 0
                    yield data_fb, data_phoneme,  target

