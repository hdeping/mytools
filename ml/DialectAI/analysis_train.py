#!/usr/bin/python

import numpy as np

from readhtk import HTKfile
#train=False
directory = "../labels/"
def getFrames(train):
    if train:
        name= directory + "label_train_list_fb.txt"
    else:
        name= directory + "label_dev_list_fb.txt"
    data = np.loadtxt(name,delimiter=' ',dtype=str)
    
    data = data[:,0]
    #print(data)
    frames = []
    i = 0
    for htk_file in data:
        htk_feature = HTKfile(htk_file)
        feature_frames = htk_feature.get_frame_num()
        i = i + 1
        frames.append(feature_frames)
        if i%1000 == 0:
            print("i = ",i)

    
    frames = np.array(frames)
    if train:
        np.savetxt("train_frames.txt",frames,fmt="%d")
    else:
        np.savetxt("dev_frames.txt",frames,fmt="%d")
def getNames(train):
    if train:
        name= directory + "label_train_list_fb.txt"
    else:
        name= directory + "label_dev_list_fb.txt"
    data = np.loadtxt(name,delimiter=' ',dtype=str)
    
    #data = data[:,0]
    #print(data)
    long_frames = []
    short_frames = []
    for htk_file in data:
        htk_feature = HTKfile(htk_file[0])
        #print(htk_feature.read_data().shape)
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
getFrames(train)
train=False
getFrames(train)
