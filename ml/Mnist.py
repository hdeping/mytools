#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-09-21 10:45:44
                    2019-10-22 20:23:38
    @project      : MNIST Data
    @version      : 1.0
    @source file  : Mnist.py

============================
"""



import numpy as np
import gzip
import struct
import PIL.Image as Image
import time
from sklearn.svm import SVC
import joblib
import pandas
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class Mnist():

    """
    Docstring for Mnist. 
    class for loading MNIST data 
    from ubyte files
    """

    def __init__(self):
        """
        TODO: to be defined1. 
        self.dirs:
            directory of MNIST data
        self.filenames:
            filenames of ubyte or ubyte.gz files
        self.train_data：
            train data of MNIST
        self.train_label：
            label of train data of MNIST
        self.test_data：
            test data of MNIST
        self.test_label：
            label of test data of MNIST
        """
        super(Mnist,self).__init__()
        self.filename = "ubyte.sh"
        self.dirs = "./data/"
        self.filenames = ["train-images-idx3-ubyte.gz",
                          "train-labels-idx1-ubyte.gz",
                          "test-images-idx3-ubyte.gz",
                          "test-labels-idx1-ubyte.gz"]

        self.train_data  = None
        self.train_label = None
        self.test_data   = None
        self.test_label  = None

    def setDirs(self,dirs):
        self.dirs = dirs 
        return 

    def die(msg):
        print(msg)
        sys.exit(1)

    def run(self):
        print(self.filenames)
        filename = self.dirs + self.filenames[0]
        fp = open(filename,'r')
        data = fp.read()
        fp.close()

        print(len(data))

        return 
    # convert ubyte file into txt format
    def convert(fname, fname_out):
        print('converting %s' % fname)
        with gzip.open(fname, 'r') as fin, open(fname_out, 'w') as fout:
            # important to read metadata in case Mr. Yann decides to change MNIST
            magic = struct.unpack('>i', fin.read(4))[0]
            if magic == 2049:
                items = struct.unpack('>i', fin.read(4))[0]
                for _ in range(items):
                    fout.write("%d\n" % ord(fin.read(1)))
            elif magic == 2051:
                images, rows, cols = struct.unpack('>3i', fin.read(12))
                imgsize = rows*cols
                for _ in range(images):
                    image = struct.unpack('%dB' % imgsize, fin.read(imgsize))
                    fout.write(" ".join(str(px-128) for px in image)+"\n")
            else:
                die('%s is not MNIST!!!' % fname)
        print('saving converted file to %s' % fname_out)
        return 

    # get data from the ubyte file
    # return: image data or label data
    def getMnistData(self,filename):
        print('load data from %s' % filename)
        results = []
        with gzip.open(filename, 'r') as fp:
            # important to read metadata in case 
            # Mr. Yann decides to change MNIST
            magic = struct.unpack('>i', fp.read(4))[0]
            if magic == 2049:
                items = struct.unpack('>i', fp.read(4))[0]
                for _ in range(items):
                    label = ord(fp.read(1))
                    results.append(label) 
                print("load label data succeed!")
                
            elif magic == 2051:
                images, rows, cols = struct.unpack('>3i', fp.read(12))
                imgsize = rows*cols
                for _ in range(images):
                    image = struct.unpack('%dB' % imgsize, fp.read(imgsize))
                    results.append(image)
                print("load image data succeed!")

            else:
                self.die('%s is not MNIST!!!' % filename)

        results = np.array(results)
        return results
    # load train and test data
    def loadData(self):
        indeces = [0,1,2,3]
        results = self.getIndexData(indeces)
        self.train_data  = results[0]
        self.train_label = results[1]
        self.test_data   = results[2]
        self.test_label  = results[3]

        return 
    def saveMnistData(self,results):
        output = ["train_data","train_label",
                  "test_data","test_label"]

        for i,data in enumerate(results):
            filename = output[i] + ".gz"
            np.savetxt(filename,data)

        print(len(self.train_data[0]))
        print(len(self.train_label))
        print(self.train_data[0][:500])
        
        print(self.train_label[:100])
        
        return 
    def train(self):
        self.loadData()

        model = SVC()
        model.fit(self.train_data,self.train_label)
        joblib.dump(model,"mnist_svc.pkl")
        test_label_predict = model.predict(self.test_data)
        num = sum(test_label_predict == self.test_lael)
        print("auc",num/len(self.test_label))
    # input: indeces, such as (2,3)
    # return :
    def getIndexData(self,indeces):
        results = []
        for index in indeces:
            filename = self.dirs + self.filenames[index]
            data = self.getMnistData(filename)
            results.append(data)
        return results
    # get train and test labels
    def getLabels(self):
        indeces = [1,3]
        results = self.getIndexData(indeces)
        self.train_label = results[0]
        self.test_label  = results[1]

    # get train and test data
    def getData(self):
        indeces = [0, 2]
        results = self.getIndexData(indeces)
        self.train_data = results[0]
        self.test_data  = results[1]

    def getTrain(self):
        indeces = [0, 1]
        results = self.getIndexData(indeces)
        self.train_data  = results[0]
        self.train_label = results[1]

    def getTest(self):
        indeces = [2, 3]
        results = self.getIndexData(indeces)
        self.test_data  = results[0]
        self.test_label = results[1]

    def showDigits(self):

        self.getTest()
        index = 1000
        image = self.test_data[index]
        print(self.test_label[index])
        indeces = np.argsort(self.test_label)
        print(self.test_label[indeces[1000:1200]])

        return
    def showAllDigits(self,type="test"):
        if type == "test":
            self.getTest()
            indeces = np.argsort(self.test_label)
        else:
            self.getTrain()
            indeces = np.argsort(self.train_label)
        slice_num = len(indeces) // 10
        index = 0
        for index in range(10):
            begin = index * slice_num
            end   = begin + slice_num
            if type == "test":
                image = self.test_data[indeces[begin:end]]
            else:
                image = self.train_data[indeces[begin:end]]
            image = self.combineImage(image)
            self.showImage(image,type,index)
            # break
        return
    # combine several images into one
    def combineImage(self,data):
        if len(data) == 1000:
            row = 20
            col = 50
        elif len(data) == 6000:
            row = 60
            col = 100
        else:
            print("Unable to combine images!!!")
            return None
        data = data.reshape((row,col,28,28))
        image = data.transpose((0,2,1,3))
        image = image.reshape((row*28,col*28,1))
        image = image.repeat(3,axis=2)

        return image
    def showImage(self,image,type,label):

        # image = np.ones(784*3,dtype=np.uint8)*255
        # image = image.reshape((28,-1,3))
        # print(image)
        # image = image.repeat(3)
        # image = image.reshape((28,-1,3))
        image = image.astype("uint8")
        digit = Image.fromarray(image)
        digit.show()
        filename = self.dirs + "%s_label%d.png"%(type,label)
        digit.save(filename)

        return

    def getDigitNum(self):
        self.getLabels()

        count = {}
        count["train"] = np.zeros(10,dtype=int)
        count["test"] = np.zeros(10,dtype=int)

        for i in self.train_label:
            count["train"][i] += 1
        for i in self.test_label:
            count["test"][i] += 1

        train = count["train"]
        test = count["test"]
        print(train)
        print(test)
        print(train + test)

        count = pandas.DataFrame(count)
        count.plot.bar(stacked = True,figsize=(12,6))
        plt.savefig("digitNum.png",dpi=200)
        plt.show()

    # how many "0"s in the train and test data


mnist = Mnist()

# mnist.loadData()
# mnist.train()
# mnist.showAllDigits(type = "test")
# mnist.showAllDigits(type = "train")
mnist.getDigitNum()