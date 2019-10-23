import numpy as np
import sys

def getDict(keyname):
    # read data from the file <keyname>
    data = np.loadtxt(keyname,delimiter=' ',dtype=str)
    dictionary = {}
    for line in data:
        dictionary[line[0]] = line[1]

    # return the dictionary
    return dictionary

def main(keyname,filename):
    # get the dictionary
    dictionary = getDict(keyname)

    # read filenames from the file <filename>
    filenames = np.loadtxt(filename,delimiter=' ',dtype=str)

    # dealing with the filename
    filename = filename.split('/')
    filename = filename[-1]
    # open a new file for output
    output = open('label_'+filename,'w')
    # deal with every line in filenames
    for name in filenames:
        # split with '/'
        line = name.split('/')
        # get the last piece
        line = line[-1]
        # split with '_'
        pre = line.split('_')
        # get the first piece
        pre = pre[0]
        print(pre,dictionary[pre])
        # store the output into a file
        output.write("%s %s\n"%(name,dictionary[pre]))

if len(sys.argv) == 1:
    print("please input two args!!!!")
    print("such as: python get_list.py lanKey.txt list_train.txt")
else:
    main(sys.argv[1],sys.argv[2])
