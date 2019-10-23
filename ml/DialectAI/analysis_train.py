#!/usr/bin/python

import numpy as np

from read_data import HTKfile
name="label_train_list_fb.txt"
data = np.loadtxt(name,delimiter=' ',dtype=str)

data = data[:,0]
#print(data)
frames = []
for htk_file in data:
    htk_feature = HTKfile(htk_file)
    feature_frames = htk_feature.get_frame_num()
    frames.append(feature_frames)

frames = np.array(frames)
np.savetxt("train_frames.txt",frames,fmt="%d")
