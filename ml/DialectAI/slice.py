import numpy as np

a = np.arange(1000)+1
#print(a)
def old():
    # (window,offset) = (10,1)
    size = 100 - 9
    b = np.zeros((size,10))
    for i in range(10):
        b[:,i] = a[i:i+size]
    print(b)
def old1():
    # (window,offset) = (10,2)
    size = (100 - 10+2)//2
    # reshape a
    c = np.reshape(a,(50,2))
    b = np.zeros((size,10))
    for i in range(5):
        start_column = 2*i
        end_column = 2*i+2
        b[:,start_column:end_column] = c[i:i+size,:]
    print(b)
def new(a,windows,stride):
    # (window,offset) = (10,2)
    length_input = len(a)
    size = (length_input - windows+stride)//stride
    # reshape a into c
    row = length_input // stride
    c = a[:stride*row]
    c = np.reshape(c,(row,stride))
    # get b
    # slice piece
    slice_num = windows // stride
    # residual piece
    res_num = windows - slice_num*stride
    b = np.zeros((size,windows))
    # the slicing part
    for i in range(slice_num):
        start_column = stride*i
        end_column = stride*(i+1)
        b[:,start_column:end_column] = c[i:i+size,:]
    # the residual part
    if res_num != 0:
        i = slice_num
        start_column = stride*slice_num
        end_column = windows
        print(i,start_column,end_column,size,c.shape)
        b[:,start_column:end_column] = c[i:,0:res_num]
    return b
def new1(a,windows,stride):
    length_input = len(a)
    # (window,offset) = (10,2)
    size = (length_input - windows+stride)//stride
    # reshape a into c
    # if length_input is not divides by stride perfectly
    row = length_input // stride + 1
    c = np.zeros(row*stride)
    c[:len(a)] = a 
    c = np.reshape(c,(row,stride))
    # get b
    # slice piece
    slice_num = windows // stride
    # residual piece
    res_num = windows - slice_num*stride
    b = np.zeros((size,windows))
    # the slicing part
    for i in range(slice_num):
        start_column = stride*i
        end_column = stride*(i+1)
        b[:,start_column:end_column] = c[i:i+size,:]
    # the residual part
    if res_num != 0:
        i = slice_num
        start_column = stride*slice_num
        end_column = windows
        #print(i,start_column,end_column,size,c.shape)
        b[:,start_column:end_column] = c[i:i+size,0:res_num]
    return b
b = new1(a,400,160)
print(b)
