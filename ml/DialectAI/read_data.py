# -*- coding:utf-8 -*-

import codecs
import copy
import random
import numpy as np

import torch

import torch.utils.data as Data
from HTKfile import HTKfile

# get the begin and end
def getBeginEnd(train,data_piece):
    piece_num = 9000
    if train:
        # train
        begin = piece_num*data_piece
        end   = piece_num*(data_piece + 1)
    else:
        # dev
        begin = 0
        end   = 3000
    return begin,end
def get_samples(file_list,train,data_piece):
    print("get samples")
    samples = 0
    max_frames = 0
    # get begin and end
    begin,end = getBeginEnd(train,data_piece)
    for i in range(begin,end):
        if i%1000 == 0:
            print(i)
        # dealing with the line
        line = file_list[i]
        # get the file name
        htk_feature = line[0]
        # get the corresponding label
        target_label = int(line[1])

        htk_file = HTKfile(htk_feature)
        feature_frames = htk_file.get_frame_num()

        max_frames = max(max_frames, feature_frames)
        samples += 1
    #samples = 3000
    return samples, max_frames


def get_data(file_list, samples, max_frames, dimension,train,data_piece):
    print("get data")
    data = torch.zeros(samples, max_frames, dimension)
    target_frames = torch.zeros(samples, 2)
    name_list = []
    # 存储数据
    line_num = 0

    # get begin and end
    begin,end = getBeginEnd(train,data_piece)
    # open the file
    for i in range(begin,end):
        if i%1000 == 0:
            print(i)
        # dealing with the line
        line = file_list[i]
        # get the file name
        htk_feature = line[0]
        # get the corresponding label
        target_label = int(line[1])

        # read the file
        htk_file = HTKfile(htk_feature)
        feature_data = htk_file.read_data()
        file_name = htk_file.get_file_name()
        feature_frames = htk_file.get_frame_num()
        
        curr_feature = torch.Tensor(feature_data)
        means = curr_feature.mean(dim=0, keepdim=True)
        curr_feature_norm = curr_feature - means.expand_as(curr_feature)
        data[line_num,:feature_frames,:] = curr_feature_norm
        target_frames[line_num] = torch.Tensor([target_label, feature_frames])
        name_list.append(file_name)
        line_num += 1

    return data, target_frames, name_list

class TorchDataSet(Data.Dataset):
#class TorchDataSet(object):
    # train or not
    def __init__(self, file_list, batch_size, chunk_num, dimension,train,data_piece):
        self._batch_size = batch_size
        self._chunck_num = chunk_num
        self._chunck_size = self._chunck_num*self._batch_size
        self._dimension = dimension

        # get data
        samples,max_frames = get_samples(file_list,train,data_piece)
        data,target_frames,name_list = get_data(file_list,samples,max_frames,dimension,train,data_piece)

        self.train_data   = data
        self.train_labels = target_frames
    def reset(self):
        random.shuffle(self.t)
    def __getitem__(self,index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]

        return img,target
    def __len__(self):
        return len(self.train_data)
