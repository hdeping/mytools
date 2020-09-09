#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-08-18 13:43:47
    @project      : control the adb
    @version      : 1.0
    @source file  : ControlADB.py

============================
"""
import os 
from easydict import EasyDict
from time import sleep
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mytools import MyCommon
from tqdm import tqdm
import pandas
from datetime import datetime



class ControlADB(MyCommon):
    """docstring for ControlADB"""
    def __init__(self):
        super(ControlADB, self).__init__()
        self.keys = {}
        self.keys["space"] = "KEYCODE_SPACE"
        self.keys["enter"] = "KEYCODE_ENTER"

        self.keys["n0"] = "KEYCODE_0"
        self.keys["n1"] = "KEYCODE_1"
        self.keys["n2"] = "KEYCODE_2"
        self.keys["n3"] = "KEYCODE_3"
        self.keys["n4"] = "KEYCODE_4"
        self.keys["n5"] = "KEYCODE_5"
        self.keys["n6"] = "KEYCODE_6"
        self.keys["n7"] = "KEYCODE_7"
        self.keys["n8"] = "KEYCODE_8"
        self.keys["n9"] = "KEYCODE_9"
        self.keys["A"] = "KEYCODE_A"
        self.keys["B"] = "KEYCODE_B"
        self.keys["C"] = "KEYCODE_C"
        self.keys["D"] = "KEYCODE_D"
        self.keys["E"] = "KEYCODE_E"
        self.keys["F"] = "KEYCODE_F"
        self.keys["G"] = "KEYCODE_G"
        self.keys["H"] = "KEYCODE_H"
        self.keys["I"] = "KEYCODE_I"
        self.keys["J"] = "KEYCODE_J"
        self.keys["K"] = "KEYCODE_K"
        self.keys["L"] = "KEYCODE_L"
        self.keys["M"] = "KEYCODE_M"
        self.keys["N"] = "KEYCODE_N"
        self.keys["O"] = "KEYCODE_O"
        self.keys["P"] = "KEYCODE_P"
        self.keys["Q"] = "KEYCODE_Q"
        self.keys["R"] = "KEYCODE_R"
        self.keys["S"] = "KEYCODE_S"
        self.keys["T"] = "KEYCODE_T"
        self.keys["U"] = "KEYCODE_U"
        self.keys["V"] = "KEYCODE_V"
        self.keys["W"] = "KEYCODE_W"
        self.keys["X"] = "KEYCODE_X"
        self.keys["Y"] = "KEYCODE_Y"
        self.keys["Z"] = "KEYCODE_Z"

        self.keys["up"]  = "KEYCODE_DPAD_UP"
        self.keys["down"]  = "KEYCODE_DPAD_DOWN"
        self.keys["left"]  = "KEYCODE_DPAD_LEFT"
        self.keys["right"]  = "KEYCODE_DPAD_RIGHT"
        self.keys["center"]  = "KEYCODE_DPAD_CENTER"
        self.keys["back"]  = "KEYCODE_DPAD_BACK"
        self.keys["home"]  = "KEYCODE_DPAD_HOME"
        self.keys["menu"]  = "KEYCODE_DPAD_MENU"

        self.adb       = "/Users/huangdeping/Library/Android/sdk/platform-tools/adb "
        self.devices = None
        if self.devices is not None:
            self.adb += "-s " + self.devices
        self.shell     = self.adb + " shell "
        self.input     = self.shell + " input "
        self.inputText = self.shell + " input text "
        self.inputKey  = self.shell + " input keyevent "

        self.keys = EasyDict(self.keys)

        self.path = "/Users/huangdeping/AndroidStudioProjects/02_adb/data/"

    def runCommands(self,commands,waiting_time=None):
        """
        docstring for runCommands
        """
        for command in commands:
            os.system(command)
            if waiting_time != None:
                sleep(waiting_time)
        return

    def testBefore(self):
        """
        docstring for testBefore
        """
        self.tap(times=14)
        self.screencap()
        self.templateMatch()
        self.back()
        self.nextScreen()
        self.collectEnergy(back=False)
        self.uiautomator(name="ui2")
        self.getAlipayCheck()
        self.analyzeUI(name="ui2.xml")
        self.getWechatCheck()
        self.analyzeAliUIs()
        self.analyzeAlipay()
        self.analyzeAlipayByJson()
        self.analyzeWechatUI(name="wechat/ui5.xml")
        self.analyzeWechatUIs()
        self.analyzeXiaomi()
        self.alipayStati()
        self.analyzeJd()
        self.wechatStati()
        self.testIsNumber()
        self.testUI()
        self.wechatRed()
        
        return
    def love(self):
        """
        docstring for love
        """
        code = "woaini"
        code = "ainidediyibaitian"
        commands = [self.inputText+code,
                    self.inputKey+self.keys.space,
                    self.inputKey+self.keys.enter]
        for i in range(20):
            
            self.runCommands(commands)

        
        return

    def swipe(self,direction="down"):
        """
        docstring for swipe
        direction = 
            down,up,left, right
        """
        left  = " 500 1000 "
        right = " 600 1000 "
        up = " 550 900 "
        down = " 550 1100 "
        command = self.input + " swipe "
        if direction == "up":
            os.system(command + up + down)
        elif direction == "down":
            os.system(command + down + up)
        elif direction == "left":
            os.system(command + left + right)
        elif direction == "right":
            os.system(command + right + left)
        else:
            print("ONLY up down left and right are valid")

        return

    def tap(self,times=1,coor=[950,2100],waiting_time=1):
        """
        docstring for tap
        """
        command = self.input + "tap %d %d"%(tuple(coor))
        for i in range(times):
            print("times: %d/%d "%(i+1,times))
            os.system(command)
            if waiting_time > 0:
                sleep(waiting_time)

        return

    def screencap(self,name="screen"):
        """
        docstring for screencap
        """
        filename = "/sdcard/%s.png"%name
        command = [self.adb + " shell screencap %s"%filename,
                   self.adb + " pull %s %s"%(filename,self.path)]
        self.runCommands(command)

        return 

    def back(self):
        """
        docstring for back
        """
        command = self.inputKey + " 4"
        os.system(command)
        return
    def nextScreen(self):
        """
        docstring for nextScreen
        """
        command = self.input + " swipe "
        up = " 550 200 "
        down = " 550 1200 "
        os.system(command + down + up)

        return

    def resizeArray(self,result,scale):
        """
        docstring for resizeArray
        array(m*b,n*b) --> array(m,n)
        """
        m,n = result.shape
        # resize the result
        m = m // scale 
        n = n // scale 

        result = result[:m*scale,:n*scale]
        result = result.reshape((m,scale,n*scale))
        result = result.transpose(0,2,1)
        result = result.reshape((m,n,scale,scale))
        result = np.average(result,axis=-1)
        result = np.average(result,axis=-1)
        return result

    def colorData(self,result):
        """
        docstring for colorData
        """
        result = self.resizeArray(result,10)
        n,m = result.shape
        x = np.linspace(0,1,m)
        y = np.linspace(0,1,n)
        plt.pcolor(x,y,result,cmap=plt.cm.jet)
        plt.colorbar()
        plt.show()
        return

    def getDist(self,arr1,arr2):
        """
        docstring for getDist
        """
        x = []
        for i,j in zip(arr1,arr2):
            x.append(i-j)
        dist = np.linalg.norm(x)
        return dist
    def templateMatch(self,template_img="energy.png",
                           target_img="screen.png",
                           threshold=[0.1,30]):
        """
        匹配图标位置，用于锁定特定图像位置
        threshold: [图像差异阈值,目标相隔像素阈值]
            模板匹配采用像素级差异作为衡量标准
            由于可能存在多个目标，通过目标之间的相隔像素作为判断不同目标的标准
        return: 返回目标中心坐标 (width,height)
        """
        tpl =cv2.imread(self.path + template_img)
        target = cv2.imread(self.path + target_img)
        methods = [cv2.TM_SQDIFF_NORMED, 
                   cv2.TM_CCORR_NORMED, 
                   cv2.TM_CCOEFF_NORMED]   
        method = methods[0]
        result = cv2.matchTemplate(target, tpl, method)
        result = (result < threshold[0])
        result = result*255 
        cv2.imwrite(self.path + "result.png",result)
        loc = np.argmax(result,axis=1)

        match = []
        for i,j in enumerate(loc):
            if j > 0:
                arr = [i,j]
                if len(match) == 0:
                    match.append(arr)
                elif self.getDist(arr,match[-1]) > threshold[1]:
                    match.append(arr)
                
        results = []
        th, tw = tpl.shape[:2]
        for line in match:
            arr = []
            arr.append(line[1]+tw//2)
            arr.append(line[0]+th//2)
            results.append(arr)
        return results

    def collectEnergy(self,back=True,index=1):
        """
        docstring for collectEnergy
        """
        self.screencap("collect")
        para ={
            # yun's settings, night
            0:["energy3.png","collect.png",[0.03,30]],
            # my settings, day
            1:["energy2.png","collect.png",[0.05,30]]
        }
        template  = para[index][0]
        target    = para[index][1]
        threshold = para[index][2] 
        targets = self.templateMatch(template_img=template,
                        target_img=target,
                        threshold=threshold)
       
        for target in targets:
            print(target)
            self.tap(coor=target)
        if back:
            self.back()
        sleep(2)
        return
    def zhifubaoTree(self,index=1):
        """
        docstring for zhifubaoTree
        """
        for i in range(9):
            self.screencap("screen")
            targets = self.templateMatch()
            for target in targets:
                print(target)
                self.tap(coor=target)
                sleep(1)
                self.collectEnergy(index=index)
            self.nextScreen()
            sleep(1)
        return

    def uiautomator(self,name="ui.xml"):
        """
        docstring for uiautomator
        """
        filename = "/sdcard/%s"%name
        command = [self.adb + " shell uiautomator  dump --compressed %s"%filename,
                   self.adb + " pull %s ."%filename]
        self.runCommands(command)
        return

    def nextScreenDist(self,distance=1000):
        """
        docstring for nextScreenDist
        """
        command = self.input + " swipe "
        up = 200
        down = up + distance
        up = " 550 %d "%up
        down = " 550 %d "%down
        os.system(command + down + up)

        return

    def bounds2Array(self,bounds):
        """
        docstring for bounds2Array
        [1,2][3,4] => [1,2,3,4]
        """
        bounds = bounds.replace("[","")
        bounds = bounds.replace("]",",")
        bounds = bounds.split(",")
        bounds.pop()
        
        return [i for i in map(int,bounds)]

    def nodeAttr(self,node,attr):
        """
        docstring for nodeAttr
        """
        if attr == "text":
            return node.getAttribute(attr)
        elif attr == "bounds":
            bounds = node.getAttribute(attr)
            bounds = self.bounds2Array(bounds)
            return bounds
        return None
    def analyzeUI(self,name="ui1.xml",bounds=False):
        """
        docstring for analyzeUI
        """
        from xml.dom.minidom import parse 

        xml = parse(name)
        nodes = xml.getElementsByTagName("node")
        results = []
        attrs = ["text","bounds"]
        for node in nodes:
            if bounds == True:
                arr = []
                bounds_arr = self.nodeAttr(node,attrs[1])
                if sum(bounds_arr) > 0:
                    text = self.nodeAttr(node,attrs[0])
                    results.append([text,bounds_arr])
            else:
                results.append(self.nodeAttr(node,attrs[0]))
        return results 
        
    def analyzeAliUI(self,name="ali/ui1.xml"):
        """
        docstring for analyzeAliUI
        """
        results = self.analyzeUI(name=name)
        lines = []
        arr   = []
        for line in results:
            if line == "":
                if len(arr) in [5,6]:
                    lines.append(arr)
                arr = []
                continue
            arr.append(line)
                
        return lines

    def analyzeAliUIs(self):
        """
        docstring for analyzeAliUIs
        """
        lines = []
        for i in tqdm(range(1,200)):
            name = "ali/ui%d.xml"%i 
            lines += self.analyzeAliUI(name=name)
        for line in lines:
            print(line)
        return
    def getScreenInfo(self,begin=1,end=1000,
                     distance=915):
        """
        docstring for getScreenInfo
        """
        for i in range(begin,end):
            print("page",i)
            self.uiautomator(name="ui%d.xml"%i)
            self.nextScreenDist(distance=distance)
        return
    def getAlipayCheck(self):
        """
        docstring for getAlipayCheck
        """
        self.getScreenInfo()
        
        return

    def getWechatCheck(self):
        """
        docstring for getWechatCheck
        """
        self.getScreenInfo(distance=950)
        return

    def analyzeAlipay(self):
        """
        docstring for analyzeAlipay
        """
        data = self.loadStrings("alipay.csv",encoding="gbk")

        lines = []
        for i,line in enumerate(data):
            line = line.split(",")
            lines.append(line[:-1])

        total = {}
        keys = lines[0][1:]
        for line in tqdm(lines[1:]):
            arr = {}
            for key,value in zip(keys,line[1:]):
                arr[key] = value
            total[line[0]] = arr 
        self.writeJson(total,"alipay.json")


        return
    def analyzeAlipayByJson(self):
        """
        docstring for analyzeAlipayByJson
        """
        data = self.loadJson("alipay.json")
        success = ['交易成功', '退款成功','还款成功', 
                 '放款成功','亲情卡付款成功','充值成功']
        state = success + ['等待付款', '交易关闭', '还款失败']
        count = {}
        money = {}
        for key in state:
            count[key] = 0
            money[key] = 0
        keys = ["交易状态","金额","收/支"]
        values = []
        for key in data:
            x = data[key][keys[0]]
            if x in success and data[key][keys[2]] == "支出":
                count[x] += 1
                y = float(data[key][keys[1]]) 
                values.append(y)
                money[x] += y
        print(count)
        print(money)
        print(max(values),min(values))
        return

    def analyzeWechatUI(self,name="wechat/ui1.xml"):
        """
        docstring for analyzeWechatUI
        """
        results = self.analyzeUI(name=name)
        
        lines = []
        arr   = []
        keywords = [":","月","日"]
        for i,line in enumerate(results):
            p = True
            for key in keywords:
                p = (p and (key in line))
            if p:
                lines.append(results[i-1:i+1])               
        return lines
    def analyzeWechatUIs(self):
        """
        docstring for analyzeAliUIs
        """
        lines = []
        for i in tqdm(range(1,193)):
            name = "wechat/ui%d.xml"%i 
            lines += self.analyzeWechatUI(name=name)
        self.writeJson(lines,"wechat.json")
        return 

    def analyzeXiaomi(self):
        """
        docstring for analyzeXiaomi
        """
        names = self.loadJson("xiaomi_files.json")
        keys = ["sleep","act","body","sport","user",
                "heart_auto","act_stage","act_minute",
                "heart"]

        # files = {}
        # for key,name in zip(keys,names):
        #     files[key] = name 
        # self.writeJson(files,"xiaomi_files.json")

        data = pandas.read_csv(names["sleep"])
        for key in data:
            print(key)
        data = np.array(data)

        sleep_hours = []
        for i,line in enumerate(data):
            hour = (line[-1] - line[-2])/60/60 
            sleep_hours.append(hour)
        plt.hist(sleep_hours,bins = 50)
        plt.savefig("sleep.png",dpi=300)
        plt.show()

        return

    def alipayStati(self):
        """
        docstring for alipayStati
        """
        keys    = ["交易对方","金额","交易状态"]
        success = ['交易成功', '退款成功','还款成功', 
                   '放款成功','亲情卡付款成功','充值成功']

        alipay = self.loadJson("alipay.json")

        stati1 = {}
        stati2 = {}
        keyword = "铁路"
        count = []
        for key in alipay:
            value = alipay[key]
            p1 = (value[keys[2]] in success)
            title = value[keys[0]]
            if title in stati1:
                stati1[title] += 1
                stati2[title] += float(value[keys[1]])
            else:
                stati1[title] = 1
                stati2[title] = float(value[keys[1]])
        stati1 = self.sortDicts(stati1)
        stati2 = self.sortDicts(stati2)
        for key in stati1:
            if keyword in key:
                print(key,stati1[key],stati2[key])
        return

    def countStati(self,array):
        """
        docstring for countStati
        array: 2d array
        statistics the number and the total values
        """
        stati1 = {}
        stati2 = {}
        for line in array:
            title = line[0]
            if title in stati1:
                stati1[title] += 1
                stati2[title] += float(line[1])
            else:
                stati1[title] = 1
                stati2[title] = float(line[1])
        stati1 = self.sortDicts(stati1)
        stati2 = self.sortDicts(stati2)
        for key in stati2:
            print(key,stati1[key],stati2[key])
        
        return stati1,stati2

    def analyzeJd(self):
        """
        docstring for analyzeJd
        """
        lines = self.loadStrings("jd/jd.csv",encoding="utf-8")

        indeces = []
        for i,line in enumerate(lines):
            if len(line) > 3 and line[-3] == ":":
                indeces.append(i)
        indeces.append(len(lines))
        total = []

        count = {}
        ignore = self.loadStrings("jd_ignore.json",encoding="utf-8")

        res = {}
        for i in range(len(indeces)-1):
            line = lines[indeces[i]:indeces[i+1]]
            new = []
            for item in line:
                if item not in ignore:
                    new.append(item)
            res[new[0]] = new[1:]
            total.append(new)
        self.writeJson(res,"jd.json")
        money = {}
        seperator = "¥"
        for line in total:
            # print(len(line),line)
            key = line[0]
            for item in line:
                if seperator in item:
                    value = item.split(seperator)[1]
                    value = float(value)
                    if key in money:
                        money[key].append(value)
                    else:
                        money[key] = [value]
        # print(money)

        stati = {}
        for i,key in enumerate(money):
            value = money[key]
            print(i,key)
            title = key[:4]
            if title in stati:
                stati[title][0] += 1
                stati[title][1] += value[0]
            else:
                stati[title] = [1,value[0]]
        for key in stati:
            print(key,stati[key])


        return

    def isNumber(self,x):
        """
        docstring for isFloat
        123.3434 -> True
        123 -> True
        23a -> False
        13.234a -> False
        """

        arr = x.split(".")
        if len(arr) in [1,2]:
            p = True
            for i in arr:
                p = (p and i.isdigit())
            return p 
        return False
    
    def getNumber(self,line):
            """
            docstring for getNumber
            """
            arr = line.split(".")[0]
            for i in range(1,len(arr)):
                if not arr[-i].isdigit():
                    break 
            return  [line[:-i-2],line[-i-2:]]
    def wechatStati1(self):
        """
        docstring for wechatStati1
        """
        data = self.loadJson("wechat.json")
        total = {}
        keys = ["+", "-", "零钱"]

        species = {}
        res = []
        year = 2020
        last_month = 19
        for i,line in enumerate(data):
            if keys[1] in line[0]:
                arr = line[0].split(keys[1])
            item = arr[:-1]
            if keys[0] in arr[-1]:
                tmp = arr[-1].split(keys[0])
                item += tmp[:-1] + [float(tmp[-1])]
            elif self.isNumber(arr[-1]):
                item += [-float(arr[-1])]
            if len(item) == 1:
                item = self.getNumber(line[0])
            item += line[1:]
            month = item[-1].split("月")[0]
            month = int(month)
            if month > last_month:
                year -= 1 
            item[-1] = "%d年%s"%(year,item[-1])
            last_month = month 
            res.append(item)
        self.writeJson(res,"wechat1.json")


        return

    def wechatStati(self):
        """
        docstring for wechatStati
        """
        data = self.loadJson("wechat1.json")

        total = []
        for line in data:
            total.append([line[0],line[-2]])
        # print(total)
        self.countStati(total)

        return 

    def testIsNumber(self):
        """
        docstring for testIsNumber
        """

        strings = [
            "234.02a",
            "23402a",
            "23402",
            "23402.",
            "234.02",
            "23,34"
            "23.02.3"
        ]
        for key in strings:
            print(key,self.isNumber(key))
        return 


    def testUI(self):
        """
        docstring for testUI
        """
        name = "ui2.xml"
        # self.uiautomator(name=name)
        data = self.analyzeUI(name=name,bounds = True)
        total = []
        for line in data:
            total.append(line[1])
        self.writeJson(total,"data.js")
        return
    def saveUIJs(self,nodes):
        """
        docstring for saveUiJs
        """
        res = []
        for node in nodes:
            res.append(node[1])
        self.writeJs(res,"dataset","ui.js")

        return 

    def getRedWindow(self,node):
        """
        docstring for getRedWindow
        """
        delta = [137,21,186-1080,-124]

        res = [i+j for i,j in zip(node,delta)]

        return res 

    def countRed(self,data):
        """
        docstring for countRed
        the number of a specific color (81,81,250)
        """
        count = 0
        height,width,_ = data.shape
        for i in range(height):
            for j in range(width):
                line = "%d%d%d"%(tuple(data[i,j]))
                if line == "8181250":
                    count += 1

        return count

    def getRed(self):
        """
        docstring for getRed
        """
        screen = 1
        ui = "wechatRed.xml"
        if screen:
            self.screencap()
            self.uiautomator(name=ui)
        image =cv2.imread(self.path + "screen.png")
        nodes = self.analyzeUI(name=ui,bounds=True)
        self.saveUIJs(nodes)
        windows = []
        for node in nodes:
            p1 = (node[1][0] == 0)
            p2 = (node[1][2] == 1080)
            height = node[1][3] - node[1][1]
            p3 = ( height > 190)
            p4 = ( height < 198)
            if  p1 and p2 and p3 and p4:
                red = self.getRedWindow(node[1])
                data = image[red[1]:red[3],red[0]:red[2]]
                x = (red[0]+red[2])/2
                y = (red[1]+red[3])/2
                line = [x,y,False]
                count = self.countRed(data)
                print(count)
                if count > 500:
                    line[-1] = True
                windows.append(line)
        return windows
    def wechatRed(self):
        """
        docstring for wechatRed
        去除微信红色消息提示
        """
        for i in range(10):
            windows = self.getRed()
            for window in windows:
                if window[-1]:
                    self.tap(coor=window[:-1])
                    self.back()
            self.nextScreenDist(distance=950)
            sleep(1)

        return

    def getUIs(self,distance=1000):
        """
        docstring for gaode
        """
        begin = 0
        end   = 120
        for i in range(begin,end):
            name = "ui%d.xml"%(i)
            self.uiautomator(name=name)
            self.nextScreenDist(distance=distance)
            nodes = self.analyzeUI(name=name)

        return

    def tvUp(self):
        """
        docstring for tvUp
        """
        
        commands = [self.inputKey + self.keys.left,
                    self.inputKey + self.keys.center
        ]
        self.runCommands(commands)
        return

    def listPackage(self):
        """
        docstring for listPackage
        """
        command = self.shell + " pm list package"
        os.system(command)
        return

    def inputStrings(self):
        """
        docstring for inputStrings
        """
        strings = "I want to go to the sky"
        strings = strings.split(" ")
        space = self.inputKey + self.keys.space 

        commands = []
        num = len(strings)
        for i in range(num-1):
            commands.append(self.inputText + strings[i])
            commands.append(space)
        commands.append(self.inputText + strings[-1])
        self.runCommands(commands)

        return
    def test(self):
        """
        docstring for test
        """
        # self.devices = "222.195.67.90:5555"
        # self.zhifubaoTree(index=1)
        # self.collectEnergy(back=False)
        # self.wechatRed()
        # self.gaode()
        # self.tvUp()
        # self.screencap()
        # self.inputStrings()

        return
