# -*- coding: UTF-8 -*-

"""

============================

    @author       : Deping Huang
    @mail address : dphuang3@iflytek.com
                    xiaohengdao@gmail.com
    @date         : 2019年09月29日 18:42:05
                    2019-10-09 14:09:15
    @project      : GUI for triangle area calculation
    @version      : 0.1
    @source file  : Triangle.py

============================
"""

import tkinter
from tkinter import messagebox
import tkinter.ttk as ttk
from .Triangle import Triangle
# from Triangle import Triangle

class MyGUI(Triangle):
    """
    """
    def __init__(self):
        super(MyGUI, self).__init__()
        self.window = tkinter.Tk()
        self.window.title("三角形面积计算器")
        self.window.geometry("480x480")
        self.lengths = []
        for i in range(3):
            self.lengths.append(tkinter.StringVar())

        self.width = 17
        self.height = 5

    def interface(self):
        texts = ["一","二","三"]
        prefix = "请输入边长"
        for i in range(3):
            self.setLabel(prefix+texts[i],i,0)
            self.setEntry(i,i,1)
        self.setButton()
        # self.width = self.width*3
        # self.height = self.height*3
        # self.setLabel("结果",3,0)
        self.setText()

        self.window.mainloop()
        return
    def setText(self):
        self.text = tkinter.Text(self.window,
                            background="lightblue",
                            width=self.width,
                            height=self.height)
        self.text.grid(row=3,column=1)
    # function for label settings
    def setLabel(self,text,row,col):
        self.label = tkinter.Label(self.window,
                                   text=text,
                                   background="lightgreen",
                                   width=self.width,
                                   height=self.height)
        if row == 3:
            self.label.grid(columnspan=3,sticky = tkinter.S)
        else:
            self.label.grid(row=row, column=col,
                            sticky=tkinter.N + tkinter.S)
        return
    # function for entry settings
    def setEntry(self,index,row,col):
        entry = tkinter.Entry(self.window,
                              textvariable=self.lengths[index],
                              width=self.width)
        entry.grid(row=row,column=col,
                   sticky=tkinter.N + tkinter.S)

        return
    # function for button settings
    def setButton(self,text=None):
        button = tkinter.Button(self.window,
                                text = "计算面积",
                                width=self.width,
                                height=self.height,
                                foreground="blue",
                                command=self.show)
        button.grid(row=0,column=2,
                    sticky=tkinter.N + tkinter.S,
                    rowspan = 3,
                    columnspan = 2)
        return
    def show(self):
        a = float(self.lengths[0].get())
        b = float(self.lengths[1].get())
        c = float(self.lengths[2].get())
        self.setLength(a,b,c)
        if self.isTriangle():
            area = self.getArea()
            text = "area is %.2f\n"%(area)
            # text = "triangle (%.2f,%.2f,%.2f), area is %.2f"%(a,b,c,area)
            # self.label.configure(text=text)
            self.text.insert(tkinter.END,text)
        else:
            text = "%.2f,%.2f,%.2f cannot construct a triangle"%(a,b,c)
            messagebox.showinfo("Error",text)

        return