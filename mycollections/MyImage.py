#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-09-15 11:21:32
    @project      : my personal image module
    @version      : 0.1
    @source file  : MyImage.py

============================
"""

import cv2

class MyImage():
    """docstring for MyImage"""
    def __init__(self):
        super(MyImage, self).__init__()
    def cutoutImage(self):
        """
        docstring for crop
        抠图程序
        扣出红色部分
        得到红底白字
        """
        path = "./"
        filename = path + "book.png"
        data = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
        image = np.zeros(data.shape)
        threshold = np.average(data[:,:,:2],axis=2)
        p = (threshold < 100)
        image[:,:,2] = 255
        image[:,:,0] = p*255
        image[:,:,1] = p*255
        cv2.imwrite(path + "book2.png",image)
        return 