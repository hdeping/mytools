#!/usr/bin/python

import numpy as np

from read_data import HTKfile
#train=False
def getFrames(train):
    if train:
        name="label_train_list_fb.txt"
    else:
        name="label_dev_list_fb.txt"
    data = np.loadtxt(name,delimiter=' ',dtype=str)
    
    data = data[:,0]
    #print(data)
    frames = []
    for htk_file in data:
        htk_feature = HTKfile(htk_file)
        feature_frames = htk_feature.get_frame_num()
        frames.append(feature_frames)
    
    frames = np.array(frames)
    if train:
        np.savetxt("train_frames.txt",frames,fmt="%d")
    else:
        np.savetxt("dev_frames.txt",frames,fmt="%d")
def getNames(train):
    if train:
        name="label_train_list_fb.txt"
    else:
        name="label_dev_list_fb.txt"
    data = np.loadtxt(name,delimiter=' ',dtype=str)
    
    #data = data[:,0]
    #print(data)
    long_frames = []
    short_frames = []
    for htk_file in data:
        htk_feature = HTKfile(htk_file[0])
        feature_frames = htk_feature.get_frame_num()
        if feature_frames < 300:
            short_frames.append(htk_file)
        else:
            long_frames.append(htk_file)
    
    long_frames = np.array(long_frames)
    short_frames = np.array(short_frames)
    if train:
        np.savetxt("train_long_fb.txt",long_frames,fmt="%s")
        np.savetxt("train_short_fb.txt",short_frames,fmt="%s")
    else:
        np.savetxt("dev_long_fb.txt",long_frames,fmt="%s")
        np.savetxt("dev_short_fb.txt",short_frames,fmt="%s")
train=True
getNames(train)
train=False
getNames(train)
