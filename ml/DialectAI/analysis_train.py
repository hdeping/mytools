#!/usr/bin/python

import numpy as np

from mydata import HTKfile
name="error.txt"
data = np.loadtxt(name,delimiter=' ',dtype=str)

#data = data[:,0]
#print(data)
frames = []
for htk_file in data:
    htk_feature = HTKfile(htk_file)
    feature_frames = htk_feature.get_frame_num()
    frames.append(feature_frames)

frames = np.array(frames)
np.savetxt("train_frames.txt",frames,fmt="%d")
