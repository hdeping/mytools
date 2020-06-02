#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-03-23 21:16:42
    @project      : doc to excel
    @version      : 0.1
    @source file  : main.py

============================
"""
# xlwt是处理excel文档输出的模块
# 可以通过pip进行安装
import xlwt
# json是处理json文件的模块
# 这是python自带的，无需额外安装
import json

class WechatDeal():
    """docstring for Deal"""
    def __init__(self):
        super(WechatDeal, self).__init__()

    # 从json文件中读取数据
    def loadJson(self, filename,encoding="utf-8"):
        """
        load data from the json file
        input: filename, string type
        return: data, dicts type
        """
        print("load data from file ",filename)
        fp = open(filename,"r",encoding=encoding)
        data = fp.read()
        if data.startswith(u'\ufeff'):
            data = data.encode('utf8')[3:].decode('utf8')
        data = json.loads(data)

        fp.close()

        return data

    def deal(self):
        """
        docstring for deal
        """
        # 从json文件中读取数据, data变量以字典形式
        # 存储数据
        filename = "好友列表.json"
        data = self.loadJson(filename)
        keys = ["nick_name", "remark_name",
                "user_name","wxid"]

        # 补全缺失字段
        # 有些数据缺少备注或者昵称
        # 用空格代替
        length = len(data)
        for i in range(length):
            for key in keys:
                if key not in data[i]:
                    data[i][key] = ""

        lines = ["昵称","备注","微信号","wxid"]

        workbook = xlwt.Workbook(encoding='utf-8')
        # 增加一个表格，命名为 好友列表
        booksheet = workbook.add_sheet("好友列表", cell_overwrite_ok=True)
        #写入第一行 "昵称","备注","微信号","wxid"
        for i in range(4):
            booksheet.write(0,i, lines[i])

        # 写入所有数据
        for i,item in enumerate(data):
            for j in range(4): 
                value = item[keys[j]]
                booksheet.write(i+1, j, value)

        # 输出到文件中
        filename = "灵江-好友列表.xls"
        print("输出表格: ",filename)
        workbook.save(filename)

        return


deal = WechatDeal()
deal.deal()