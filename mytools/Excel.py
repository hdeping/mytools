#!/usr/local/bin/python3
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-04-15 09:24:45
                    2019年09月23日 23:15:16
                    2019-10-08 09:42:00
                    2019年10月08日 10:03:38
    @project      : tian di jing hua
    @version      : 1.5
    @source file  : function.py

============================
"""


import xlrd
from xlrd import xldate_as_tuple
import xlwt
import numpy as np
import time
import json
from .MyCommon import MyCommon


class Excel(MyCommon):
    """
    Dealing with Excel files with xlrd and xlwt module
    Some functions are inheritated from MyCommon
    """
    def __init__(self):
        """
        self.dirs:
            the directory where we read and save our data
        self.total_data:
            parameter for the excel data
        """
        super(Excel, self).__init__()
        self.dirs = "D:\\code\\圆圆\\"

    def loadData(self):
        """
        load data from self.filename,
        and get self.total_data
        """
        print(self.filename + " opened")
        self.total_data = xlrd.open_workbook(self.filename)

        return

    def init(self,filename):
        """
        set the filename and get self.total_data
        from the file
        """
        self.setFilename(filename)
        self.loadData()
        return
    def getSheets(self):
        """
        get three arrays from three sheets of 
        the two xlsx files
        """
        t1 = time.time()
        filename = "天地精华6月账单7.11.xlsx"
        self.init(filename)
        # 商家出库单号，运费
        # 第四列，倒数第一列
        Sheet1 = self.table2Array("账单")
        sheet_name = "Sheet2"
        # 外部单号，京东C端运费
        # 第三列，倒数第三列
        Sheet2 = self.table2Array(sheet_name)

        filename = "6月原始数据-jd.xlsx"
        self.init(filename)
        # 原始单号，运费
        # 第二列，倒数第二列
        Sheet3 = self.table2Array("6月原始数据")
        t2 = time.time()
        print("time: ", t2 - t1)

        print(Sheet1.shape, Sheet2.shape, Sheet3.shape)
        self.Sheet1 = Sheet1
        self.Sheet2 = Sheet2
        self.Sheet3 = Sheet3

        return
    def getMatch(self):
        """
        compare three sheets and get the data
        with common keys
        """
        dict1 = self.getFeeDicts(self.Sheet1, 3, -2)
        dict2 = self.getFeeDicts(self.Sheet2, 2, -3)
        dict3 = self.getFeeDicts(self.Sheet3, 1, -2)
        self.writeCommon(dict1, dict2, "result12.json")
        self.writeCommon(dict1, dict3, "result13.json")
        self.writeCommon(dict2, dict3, "result23.json")

        return


    def getNewDict(self, filename):
        """
        get rid of "\t" in the key of a dictionary
        read dicts from the filename
        print the new dicts to a new file
        """
        dicts = self.loadJson(filename)
        res = {}
        for key in dicts:
            if key == "":
                continue
            value = dicts[key]
            key = key.replace("\t","")
            res[key] = value
        self.writeJson(res, "new_%s" % (filename))
        return
    def writeCommon(self, dict1, dict2, filename):
        """
        write data with common keys into a file
        input: two dicts and a filename
        """
        res = self.getCommon(dict1, dict2)
        self.writeJson(res, filename)
        return

    def getFeeDicts(self, sheet, i, j):
        """
        get a dict from two arrays
        such as , ["a","b"], [1,2] --->
        {"a":1,"b":2}
        intput: sheet, a 2D array
                i,j, index of two columns of the sheet
        """
        keys = sheet[1:,i]
        values = sheet[1:,j]
        res = self.getDicts(keys, values)

        return res
    def table2Array(self, sheet_name):
        """
        把表格转化为数组形式
        input: sheet_name, name of the sheet

        data in the sheet_name would be extracted
        to a array from the self.total_data
        """

        data = []
        origin_table = self.total_data.sheet_by_name(sheet_name)

        rows = origin_table.nrows
        cols = origin_table.ncols

        print(rows,cols)
        # 更新时间
        # 配送门店

        for i in range(rows):
            rows_data = []
            for j in range(cols):
                value = origin_table.cell(i,j).value
                rows_data.append(value)

            data.append(rows_data)

        data = np.array(data)

        return data

    def writeXlsx(self, table_array, filename, sheet_name):
        """
        把数组输出成xls格式文件
        暂时不支持输出xlsx格式，必要的话
        暂时可用excel将xls转化成xlsx
        input: table_array, array type
               filename, name of the xls file
               sheet_name, name of the sheet
        """

        workbook = xlwt.Workbook(encoding='utf-8')
        booksheet = workbook.add_sheet(sheet_name, cell_overwrite_ok=True)

        rows = len(table_array)
        cols = len(table_array[0])
        print("表格大小： %d 行, %d 列"%(rows,cols))
        for i in range(rows):
            for j in range(cols):
                # print('index',i,j)
                booksheet.write(i,j,table_array[i][j])

        workbook.save(filename)
        return

    def writeDictsXlsx(self):
        """
        load dicts from json files and 
        write array into a xls file
        names = [["商家出库单号","运费"],
                 ["外部单号","京东C端运费"],
                 ["原始单号","运费"],
                 ["2与1对比单号","2减去1运费差额"],
                 ["3与1对比单号","3减去1运费差额"],
                 ["3与2对比单号","3减去2运费差额"]]
        """

        sheet_name = "汇总"
        workbook = xlwt.Workbook(encoding='utf-8')
        booksheet = workbook.add_sheet(sheet_name, cell_overwrite_ok=True)

        dicts = []
        dicts.append(self.loadJson("new_dict1.json"))
        dicts.append(self.loadJson("new_dict2.json"))
        dicts.append(self.loadJson("new_dict3.json"))
        dicts.append(self.getCommon(dicts[0], dicts[1]))
        dicts.append(self.getCommon(dicts[0], dicts[2]))
        dicts.append(self.getCommon(dicts[1], dicts[2]))
        names = [["商家出库单号","运费"],
                 ["外部单号","京东C端运费"],
                 ["原始单号","运费"],
                 ["2与1对比单号","2减去1运费差额"],
                 ["3与1对比单号","3减去1运费差额"],
                 ["3与2对比单号","3减去2运费差额"]]

        for i in range(3,6):
            index = (i - 3)*2
            name = names[i]
            print(name,index,len(dicts[i]))
            self.writeSheet(booksheet, dicts[i], name, index)

        filename = "运费对比结果.xls"
        workbook.save(filename)

        return

    def writeSheet(self, booksheet, dicts, names, index):
        """
        booksheet: class of the sheet of the excel
        dicts: data
        names: array, ["name1","name2"]
        index: index of the column
        """

        booksheet.write(0,index, names[0])
        booksheet.write(0,index + 1, names[1])
        for i,key in enumerate(dicts):
            value = dicts[key]
            booksheet.write(i+1, index, key)
            booksheet.write(i+1, index + 1, value)

        return

    def getStoreKeys(self, data, first_row_dicts):
        """
        得到所有配送门店类型
        """

        res = []
        store_col = first_row_dicts['配送门店']
        for i in range(1,len(data)):
            key = data[i][store_col]
            if key not in res:
                res.append(key)
        return res

    def splitData(self, data, first_row_dicts, dicts_species):
        """
        将原始数据数组进行拆分
        得到一个字典
        键是配送门店名称，
        值是 二维数据
        每一行是一个订单
        每一列是相应订单的品相，日期，数量
        """
        new_data_dicts = {}
        # split data into a dict
        # with stores id as keys
        keys = self.getStoreKeys(data,first_row_dicts)
        for key in keys:
            new_data_dicts[key] = []
        # ignore the first row
        # 货品名称，更新时间，数量
        store_col = first_row_dicts['配送门店']
        specie_col = first_row_dicts['货品名称']
        update_col = first_row_dicts['更新时间']
        number_col = first_row_dicts['数量']

        all_species = []
        for line in data[:-1]:
            stati_line = []
            specie = line[specie_col]
            # ignore the keys not in dicts_species
            if specie not in dicts_species:
                continue
            specie = dicts_species[specie]
            # get all the species
            if specie not in all_species:
                all_species.append(specie)
            stati_line.append(specie)
            stati_line.append(line[update_col])
            number = float(line[number_col])
            number = int(number)
            stati_line.append(number)
            new_data_dicts[line[store_col]].append(stati_line)

        all_species.sort()
        return new_data_dicts,all_species

    def getFirstRowDicts(data):
        """
        得到第一行（更新时间，数量等关键字）对应的列数
        返回一个字典
         比如  res['订单'] = 0等信息
        """
        res = {}
        for i,line in enumerate(data[0]):
            res[line] = i
        return res

    def getFoundamentalDicts(self, data):
        """
        基础数据中
        货品名称和品相的对应关系
        返回一个字典
        """
        res = {}

        for line in data:
            key = line[0].replace('\t','')
            value = line[1]
            res[key] = value

        return res
    def arr2Dicts(self, arr):
        """
        一个一维字符串数组转为字典
        """
        res = {}
        for i,line in enumerate(arr):
            res[line] = i
        return res

    def getDicts(self, keys, values):
        """
        keys, values ---> dicts
        """
        res = {}
        for key,value in zip(keys,values):
            res[key] = float(value)
        return res

    def getStatiData(self, splitted_data, all_species, dates):
        """
        输入:
            splitted_data: 分割后的数据
            all_species  : 品相
            dates        : 日期，从"1日"到"31日"
        """

        # get rows dicts (species)
        rows_dicts = arr_to_dicts(all_species)
        cols_dicts = arr_to_dicts(dates)
        # print(rows_dicts)
        # print(cols_dicts)

        rows = len(all_species)
        cols = len(dates)

        stati_data = {}
        for key in splitted_data:
            table = splitted_data[key]
            data = np.zeros((rows,cols),dtype=int)
            for line in table:
                i = rows_dicts[line[0]]
                j = cols_dicts[line[1]]
                data[i,j] += line[2]
            stati_data[key] = data

        # get total data
        data = np.zeros((rows,cols),dtype=int)
        for key in stati_data: # wrong here!!!!!
            value = stati_data[key]
            data = data + value
        stati_data['汇总'] = data

        return stati_data

    def writeSheetOld(self, workbook, key, data):
        """
        写入一个sheet
        输入:
              workbook: excel类
              key     : sheet名称
              data    : 待写入数组
        """
        booksheet = workbook.add_sheet(key, cell_overwrite_ok=True)

        # 原始数据
        rows = len(data)
        cols = len(data[0])
        print("原始数据:")
        print("        表格大小： %d 行, %d 列"%(rows,cols))
        for i in range(rows):
            for j in range(cols):
                # print('index',i,j)
                booksheet.write(i,j,data[i][j])

    def writeSpeciesSheet(self, workbook, key, data, rowsAndCols):
        """
        写入一个sheet，记录统计信息
        输入:
              workbook   : excel类
              key        : sheet名称，这里是一个配送门店
              data       : 待写入数组，大小是(13,31)
              rowsAndCols: 数组，第一行是品相数组，长度为13
                                第二行是日期数组，长度为31

        格式:
        门店名称 品相 1日 2日。。。。。 31日  合计配送数量
                。。。。。。。。。。。。。。。。。。。。
                合计
        """

        booksheet = workbook.add_sheet(key, cell_overwrite_ok=True)


        rows = len(data)
        cols = len(data[0])

        # write the key in the first unit cell
        booksheet.write(0,0,key)
        booksheet.write(0,1,"品相")
        booksheet.write(rows+1,1,"合计")
        booksheet.write(0,cols+2,"合计配送数量")
        for i in range(rows):
            booksheet.write(i+1,1,rowsAndCols[0][i])
        for i in range(cols):
            booksheet.write(0,i+2,rowsAndCols[1][i])

        print("配送门店: %s"%(key))
        print("         表格大小： %d 行, %d 列"%(rows,cols))
        for i in range(rows):
            for j in range(cols):
                # print('index',i,j)
                booksheet.write(i+1,j+2,int(data[i][j]))
        # stati
        sum_rows = np.sum(data,axis = 1)
        sum_cols = np.sum(data,axis = 0)
        for i in range(rows):
            booksheet.write(i+1,cols+2,int(sum_rows[i]))
        for i in range(cols):
            booksheet.write(rows+1,i+2,int(sum_cols[i]))
        # sum of all
        booksheet.write(rows+1,cols+2,int(np.sum(data)))

        return

    def writeTotalXls(self, data, final_data, filename, rowsAndCols):
        """
        所有内容输出成一个xls格式excel文件:
        输入:
              data       : 原始数据数组,大小是 (3225,29)
              final_data : 处理后的数组字典
              filename   : 输出文件名称，比如 "final.xls"
              rowsAndCols: 数组，第一行是品相数组，长度为13
                                第二行是日期数组，长度为31
        """

        workbook = xlwt.Workbook(encoding='utf-8')

        # 原始数据
        self.writeSheet(workbook,'原始数据',data)

        # 处理后的数据
        count = 0
        for key in final_data:
            count += 1
            print("门店",count)
            self.writeSpeciesSheet(workbook,key,final_data[key],rowsAndCols)

        workbook.save(filename)

        return