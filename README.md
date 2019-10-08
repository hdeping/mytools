[TOC]

# Introduction to mytools module

my own tools for gui, math calculations and so on. If you have a linux or Macos environment, you should move the mytools
directory to a directory within PYTHONPATH(you can edit it in a .bashrc
or .zshrc file). If you have a windows environment, you can edit PYTHONPATH
following the guide as it shows in the link: 
https://blog.csdn.net/Tona_ZM/article/details/79463284

# class MyCommon

## Introduction

It is a collection of some common used functions, I pack them into a class called MyCommon. Some functions such as setDirs, loadJson, writeJson are highly-frequent used,  someone can inheritate this class if someone want to use the functions.

A simple example was displayed below:

```python
from mytools import MyCommon
test = MyCommon()
filename = "test.txt"
data = test.loadJson(filename)
```

## Methods

Latestly, there are 7 kinds of methods in the class. They are introduced in the following contexts.

### setDirs

Input: a relative or absolute path,  string type

Return: None, but self.dirs is changed

The parameter self.dirs would be reset in the function

### setFilename

Input: a file name, string type

Return: None, but self.filename is changed

The parameter self.filename would be reset by the full path in the function.

### setFileDirs

Input: a full path of the file name, string type

Return: None, but self.filename is changed

The parameter self.filename would be reset by the full path in the function.

### getCommon

Input: Two dicts with dicts type

Return: a dictionary with dicts type

A dictionary with common keys as keys and  the value-differences as values was returned in the function. For example:

```python
A = {"a":10,"b":20}
B = {"a":20,"c":10}
C = self.getCommon(A,B)
# C would be {"a":10}
```



### writeJson

Input: data with dicts type, and a filename with string type

Return: None, but data was written into a file with json type	

### loadJson

Input: a filename with string type

Return: data with dicts type

### writeCSV

Input: data with array type, and a filename with string type

Return: None, but data was written into a  file with csv format

# class Excel

## Introdution

This class was based on the python module xlrd and xlwt. It is used for manipulate the excel files. Usually, wps or office excel could be very slow,  if the xls file is a little bit large(such as 100M Bytes).  Furthermore, when we want do a little bit advanced operations to our data, . This class would make our operations much more quick, even we do not have a good PC comfiguration.

A simple example was displayed below:

```python
from mytools import Excel
excel = Excel()
excel.writeDictsXlsx()
```

# class Triangle 

## Introduction

It is a set of functions for triangle computations. In this module , you can get the radius and the center coordinate of the inscribed, escribed as well as circumscribed circles of a triangle when the coordinates of the vertices were given. Furthermore, you can get the line equations of the middle lines, orthogonal lines, middle orthogonal lines or bisection lines. There are 44 methods in this module.

A simple example was displayed below:

```python
from mytools import Triangle
triangle = Triangle()
triangle.run()
```

After running the above code, an image like this would be generated:

![triangle](figures/triangle.png)



# class TurtlePlay

## User Guide

### Draw Squares

It is easy to use TurtlePlay, for example:
```python
from mytools imort TurtlePlay
play = TurtlePlay()
play.test()
```
After running the above code, you will see an animation.

Finally,  we will get a figure as the following one shows:

![test](figures/test.png)

### Draw Polygons

It is easy to use TurtlePlay, for example:

```python
from mytools imort TurtlePlay
play = TurtlePlay()
play.testPolygon()
```

After running the above code, you will see polygons are drawn one after another.

Finally,  we will get a figure as the following one shows. (Note that there is a polygon for each unit)

![polygon5](figures/polygon5.png)

# Class NameAll

## User Guide

It is easy to use class NameAll, for example:

```python
from mytools imort NameAll
test = NameAll()
test.run()
```

# Class DrawPig

This code is modified from a blog(https://www.cnblogs.com/nowgood/p/turtle.html#_nav_11, [@](https://www.cnblogs.com/nowgood/p/turtle.html#4084602) 江城青椒肉丝), I just change the code into a moduled one, it would be more easy-reading and understanding.

It is used for drawing a cut pig, you should use it with the following code:

```python
from mytools imort DrawPig
pig = DrawPig()
pig.cutePig()
```

Finally, you can get a cute pig as it shows below:

![pig](figures/pig.png)

