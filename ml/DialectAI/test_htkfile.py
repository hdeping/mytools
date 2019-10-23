# -*- coding:utf-8 -*-

import codecs
import copy
import random

from HTKfile import HTKfile
htk_feature = "/home/ncl/hdp/18_aichallenge/aichallenge/train/1.getFB40/fb40/minnan_train_speaker12_047.fb"

htk_file = HTKfile(htk_feature)
#print("start frame num is ",htk_file.get_start_frame())
#print("end frame num is ",htk_file.get_end_frame())
#print("end frame num is ",htk_file.get_end_frame())
#print("file name is",htk_file.get_file_name())
print("start frame",            htk_file.get_start_frame()       )
print("end frame",              htk_file.get_end_frame()         )
print("frame num",              htk_file.get_frame_num()         )
print("sample period",          htk_file.get_sample_period()     )
print("bytes of one frame",     htk_file.get_bytes_of_one_frame())
print("file name",              htk_file.get_file_name()         )
print("feature dim",            htk_file.get_feature_dim()       )
#print("state label",            htk_file.get_state_label()         )

# read data from the htk_file
data = htk_file.read_data()
print(data.shape)
