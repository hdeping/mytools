
from readpcm import pcmdata
filename="/home/ncl/hdp/18_aichallenge/data/nanchang/dev/speaker31/short/nanchang_dev_speaker31_092.pcm"


pcm_data = pcmdata(filename,400,160)

arr = pcm_data.read_data()
print(arr.shape)
