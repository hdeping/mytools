from readhtk import HTKfile
import numpy as np

def print_shape(name):
    #0.800000,4.180000,3.380000,4.560000
    htk = HTKfile(name)
    data = htk.read_data()
    shape1 = data.shape
    name = name.replace('fb40','fb40_noVAD')
    htk = HTKfile(name)
    data = htk.read_data()
    shape2 = data.shape
    size1 = shape1[0]
    size2 = shape2[0]
    return size1,size2
filename = '../1.getFB40/list_dev.txt'
names= np.loadtxt(filename,delimiter=' ',dtype=str)

file_dir = '/home/ncl/hdp/18_aichallenge/aichallenge/train/1.getFB40/fb40/'
i = 0
output = []
for name in names:
    # split with '/'
    name = name.split('/')
    # get the last item
    name = name[-1]
    # replace to fb
    name = name.replace('pcm','fb')
    # get the full path of the fb file
    name = file_dir + name
    size1,size2 = print_shape(name)
    i = i+1
    print(i,size1,size2)
    output.append([size1,size2])
output = np.array(output)

outputname = 'fb_shape.txt'
np.savetxt(outputname,output,delimiter=',')
