#!/usr/local/bin/python3

"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-04-15 09:22:30
                    2019年09月24日 00:09:35
                    2019-10-08 09:42:14
    @project      : tian di jing hua
    @version      : 1.5
    @source file  : main.py

============================
"""

# 从另一个文件function.py导入函数
# 每一个函数完成一个特定的功能
from Excel import Excel
# 导入excel文件读取模块
import xlrd 
# 导入数组处理模块
import numpy as np
import time


# 原始文件，包含基础数据和原始数据的表格

excel = Excel()
# excel.get_new_dict("dict1.json")
# excel.get_new_dict("dict2.json")
# excel.get_new_dict("dict3.json")
excel.writeDictsXlsx()
# excel.get_sheets()
# excel.get_match()
