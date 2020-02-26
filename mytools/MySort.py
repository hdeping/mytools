#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-02-26 14:03:59
    @project      : my practice for sorting algorithms
    @version      : 0.1
    @source file  : MySort.py

============================
"""

class MySort():
    """
    my practice for sorting algorithms """
    def __init__(self):
        super(MySort, self).__init__()

        return

    def mergeSort(self,arr):
        """
        docstring for mergeSort
        arr:
            array 
        """
        if len(arr) == 1:
            return arr 
        elif len(arr) == 2:
            m,n = arr 
            if m > n:
                m,n = n,m 
            return [m,n]
        else:
            num = len(arr) // 2 
            res = []

            arr1 = self.mergeSort(arr[:num])
            arr2 = self.mergeSort(arr[num:])
            count = 0
            print(arr1,arr2)
            for index,i in enumerate(arr2):
                if count == num:
                    break
                tmp = arr1[count:].copy()
                for j in tmp:
                    if i < j:
                        res.append(i)
                        break
                    else:
                        res.append(j)
                        count += 1
            if count == num:
                if index < len(arr2) - 1:
                    index = index - 1
                res += arr2[index:]
            else:
                res += arr1[count:]

            return res
    def test(self):
        """
        docstring for test
        """
        arr = [3,5,9,1,10,200,2000,-9,20,50,60,55,11]
        # arr = [3,5,9,1,10,200,2000,-9]
        # arr = [3,5,9,1,3,1,1,1]
        # arr = [3,9,7]
        print(arr)
        arr = self.mergeSort(arr)
        print(arr)

        return 

mySort = MySort()
mySort.test()
        