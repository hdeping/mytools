Help on package mytools:

NAME
    mytools - ============================

DESCRIPTION
        @author       : Deping Huang
        @mail address : xiaohengdao@gmail.com
        @date         : 2019-10-07 00:27:08
        @project      : my python tools
        @version      : 1.0
        @source file  : __init__.py
    
    ============================

PACKAGE CONTENTS
    DistanceMeasure
    DrawCurve
    DrawPig
    Excel
    GetDoc
    MyCommon
    MyGUI
    NameAll
    Triangle
    TurtlePlay

FILE
    /Users/huangdeping/c/02_python/44_mytools/mytools/__init__.py


Help on class DrawCurve in mytools:

mytools.DrawCurve = class DrawCurve(builtins.object)
 |  docstring for DrawCurve
 |  It is used for visualization of scientific data
 |  
 |  Methods defined here:
 |  
 |  __init__(self)
 |      self.lineList:
 |          list for the line types
 |      self.typeNum:
 |          length of self.lineList
 |      self.colorList:
 |          list for the colors
 |      self.colorNum:
 |          length of self.colorList
 |      
 |      the line type:
 |          o: circle point
 |          p: pegagon point
 |          h: hexagon point
 |          ^: triangal point(angle up)
 |          v: triangal point(angle down)
 |          >: triangal point(angle right)
 |          <: triangal point(angle left)
 |          -: solid line
 |          --: dash line
 |  
 |  barsData(self, filename)
 |      visual the data with bars
 |  
 |  draw(self)
 |      function for drawing
 |  
 |  getFileName(self)
 |      get the filename of the output png 
 |      corresponding to how many images exists in the 
 |      current directory
 |  
 |  histData()
 |      get a histagram
 |  
 |  plotData(self)
 |      load data from self.dataFileName and 
 |      plot the data with lines
 |  
 |  setCurveLabels(self)
 |      setup for the labels of the curve
 |  
 |  setCurveLegends(self)
 |      setup for the legends of the curve
 |  
 |  setCurveTicks(self)
 |      setup for the ticks of the curve
 |  
 |  setCurveTitle(self)
 |      setup for the title of the curve
 |  
 |  setDataFileName(self, dataFileName)
 |      setup for the dataFileName
 |  
 |  setSizes(self, titleSize, labelSize, tickSize)
 |      setup for the fontsizes of title, labels
 |      and ticks
 |  
 |  splineData(filename)
 |      get the B-spline of the scatter point
 |  
 |  test(self)
 |      test for the DrawCurve module
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)

Help on class DrawPig in mytools:

mytools.DrawPig = class DrawPig(mytools.TurtlePlay.TurtlePlay)
 |  docstring for DrawPig
 |  
 |  Method resolution order:
 |      DrawPig
 |      mytools.TurtlePlay.TurtlePlay
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __init__(self)
 |      self.theta: 
 |          rotation angle for each time
 |      self.speed:
 |          frames per second
 |          such as : self.speed = 4
 |          the panel will be drawed 4 times in a second
 |      self.length:
 |          length of the line
 |      self.number:
 |          cycles number 
 |      
 |      All the above parameters were initialized to be None
 |  
 |  cutePig(self)
 |  
 |  drawBody(self, height, centers)
 |  
 |  drawCircles(self, centers)
 |  
 |  drawLines(self, lines)
 |      input: lines, 2d array
 |      such as : [[0,90],[90,2]...]
 |  
 |  drawPigBody(self)
 |  
 |  drawPigCheek(self)
 |  
 |  drawPigEar(self, lines, centers)
 |  
 |  drawPigEars(self)
 |  
 |  drawPigEye(self, lines)
 |  
 |  drawPigEyes(self)
 |  
 |  drawPigFeet(self)
 |  
 |  drawPigFoot(self, lines)
 |  
 |  drawPigHands(self)
 |  
 |  drawPigHead(self)
 |  
 |  drawPigInit(self)
 |  
 |  drawPigMouth(self)
 |  
 |  drawPigNose(self, lines)
 |  
 |  drawPigNoses(self)
 |  
 |  drawPigTail(self)
 |  
 |  fillEar(self, centers)
 |  
 |  fillRegion(self, radius, color=None)
 |  
 |  setPigColors(self)
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from mytools.TurtlePlay.TurtlePlay:
 |  
 |  draw(self)
 |      draw the pattern
 |      length is smaller and smaller in each step
 |  
 |  drawPolygons(self, n)
 |      draw polygons 
 |      add length by 3.5 after drawing a polygon
 |  
 |  drawSquares(self)
 |      draw squares
 |      add length by 3.5 after drawing a square
 |  
 |  initParas(self)
 |      initialize the parameters
 |  
 |  polygon(self, n)
 |      draw a single polygon
 |      with the shape turtle
 |  
 |  printParas(self)
 |      print out all the parameters
 |      including theta, length, number and speed
 |  
 |  setColor(self, colors)
 |      set the color 
 |      input: colors, array with two items
 |      such as : ["red","red"]
 |  
 |  setLength(self, length)
 |      set the length of the line
 |  
 |  setNumber(self, number)
 |      set the number of cycles
 |  
 |  setParas(self, theta, length, number, speed)
 |      setup for all the parameters
 |      including theta, length, number and speed
 |  
 |  setSpeed(self, speed)
 |      set the fps(frames per second)
 |  
 |  setTheta(self, theta)
 |      set the rotation angle
 |  
 |  square(self)
 |      draw a single square
 |      with the shape turtle
 |  
 |  test(self)
 |      test for drawing the squares
 |  
 |  testPolygon(self)
 |      test for drawing the polygons
 |      n was set to be 5
 |  
 |  writeImage(self)
 |      write the image into ps and pdf
 |      ps2pdf is a linux command which 
 |      converts a .ps file into a .pdf one
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from mytools.TurtlePlay.TurtlePlay:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)

Help on class Excel in mytools:

mytools.Excel = class Excel(mytools.MyCommon.MyCommon)
 |  Dealing with Excel files with xlrd and xlwt module
 |  Some functions are inheritated from MyCommon
 |  
 |  Method resolution order:
 |      Excel
 |      mytools.MyCommon.MyCommon
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __init__(self)
 |      self.dirs:
 |          the directory where we read and save our data
 |      self.total_data:
 |          parameter for the excel data
 |  
 |  arr2Dicts(self, arr)
 |      一个一维字符串数组转为字典
 |  
 |  getDicts(self, keys, values)
 |      keys, values ---> dicts
 |  
 |  getFeeDicts(self, sheet, i, j)
 |      get a dict from two arrays
 |      such as , ["a","b"], [1,2] --->
 |      {"a":1,"b":2}
 |      intput: sheet, a 2D array
 |              i,j, index of two columns of the sheet
 |  
 |  getFirstRowDicts(data)
 |      得到第一行（更新时间，数量等关键字）对应的列数
 |      返回一个字典
 |       比如  res['订单'] = 0等信息
 |  
 |  getFoundamentalDicts(self, data)
 |      基础数据中
 |      货品名称和品相的对应关系
 |      返回一个字典
 |  
 |  getMatch(self)
 |      compare three sheets and get the data
 |      with common keys
 |  
 |  getNewDict(self, filename)
 |      get rid of "    " in the key of a dictionary
 |      read dicts from the filename
 |      print the new dicts to a new file
 |  
 |  getSheets(self)
 |      get three arrays from three sheets of 
 |      the two xlsx files
 |  
 |  getStatiData(self, splitted_data, all_species, dates)
 |      输入:
 |          splitted_data: 分割后的数据
 |          all_species  : 品相
 |          dates        : 日期，从"1日"到"31日"
 |  
 |  getStoreKeys(self, data, first_row_dicts)
 |      得到所有配送门店类型
 |  
 |  init(self, filename)
 |      set the filename and get self.total_data
 |      from the file
 |  
 |  loadData(self)
 |      load data from self.filename,
 |      and get self.total_data
 |  
 |  splitData(self, data, first_row_dicts, dicts_species)
 |      将原始数据数组进行拆分
 |      得到一个字典
 |      键是配送门店名称，
 |      值是 二维数据
 |      每一行是一个订单
 |      每一列是相应订单的品相，日期，数量
 |  
 |  table2Array(self, sheet_name)
 |      把表格转化为数组形式
 |      input: sheet_name, name of the sheet
 |      
 |      data in the sheet_name would be extracted
 |      to a array from the self.total_data
 |  
 |  writeCommon(self, dict1, dict2, filename)
 |      write data with common keys into a file
 |      input: two dicts and a filename
 |  
 |  writeDictsXlsx(self)
 |      load dicts from json files and 
 |      write array into a xls file
 |      names = [["商家出库单号","运费"],
 |               ["外部单号","京东C端运费"],
 |               ["原始单号","运费"],
 |               ["2与1对比单号","2减去1运费差额"],
 |               ["3与1对比单号","3减去1运费差额"],
 |               ["3与2对比单号","3减去2运费差额"]]
 |  
 |  writeSheet(self, booksheet, dicts, names, index)
 |      booksheet: class of the sheet of the excel
 |      dicts: data
 |      names: array, ["name1","name2"]
 |      index: index of the column
 |  
 |  writeSheetOld(self, workbook, key, data)
 |      写入一个sheet
 |      输入:
 |            workbook: excel类
 |            key     : sheet名称
 |            data    : 待写入数组
 |  
 |  writeSpeciesSheet(self, workbook, key, data, rowsAndCols)
 |      写入一个sheet，记录统计信息
 |      输入:
 |            workbook   : excel类
 |            key        : sheet名称，这里是一个配送门店
 |            data       : 待写入数组，大小是(13,31)
 |            rowsAndCols: 数组，第一行是品相数组，长度为13
 |                              第二行是日期数组，长度为31
 |      
 |      格式:
 |      门店名称 品相 1日 2日。。。。。 31日  合计配送数量
 |              。。。。。。。。。。。。。。。。。。。。
 |              合计
 |  
 |  writeTotalXls(self, data, final_data, filename, rowsAndCols)
 |      所有内容输出成一个xls格式excel文件:
 |      输入:
 |            data       : 原始数据数组,大小是 (3225,29)
 |            final_data : 处理后的数组字典
 |            filename   : 输出文件名称，比如 "final.xls"
 |            rowsAndCols: 数组，第一行是品相数组，长度为13
 |                              第二行是日期数组，长度为31
 |  
 |  writeXlsx(self, table_array, filename, sheet_name)
 |      把数组输出成xls格式文件
 |      暂时不支持输出xlsx格式，必要的话
 |      暂时可用excel将xls转化成xlsx
 |      input: table_array, array type
 |             filename, name of the xls file
 |             sheet_name, name of the sheet
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from mytools.MyCommon.MyCommon:
 |  
 |  getCommon(self, dict1, dict2)
 |      get the common key of the two dicts
 |      and get a dicts with the difference value
 |      as a new value.
 |  
 |  loadFile(self, filename)
 |      load data from the file
 |      If it is a yml or yaml type,
 |      yaml module would be used,
 |      if it is a json type, json module
 |      would be used, any other formats are not 
 |      supported
 |      input: filename, string type
 |      return: data, dicts type
 |  
 |  loadJson(self, filename)
 |      load data from the json file
 |      input: filename, string type
 |      return: data, dicts type
 |  
 |  setDirs(self, dirs)
 |      setup for the data directory
 |  
 |  setFileDirs(self, filename)
 |      setup for the filename with absolute path
 |      input: filename, such as "/home/test/data.txt"
 |  
 |  setFilename(self, filename)
 |      setup for the filename with relative path
 |      input: filename, such as "data.txt"
 |  
 |  writeCSV(self, table_array, filename)
 |      把数组写成csv文件输出，以tab符作为间隔符号
 |      同时，应去除原始文本中多余的tab符
 |      
 |      write the array into a csv file with the tab as 
 |      delimiters. On the same time, extraordinary tabs 
 |      are deliminated.
 |  
 |  writeFile(self, data, filename)
 |      write data to the file.
 |      If it is a yml or yaml type,
 |      yaml module would be used,
 |      if it is a json type, json module
 |      would be used, any other formats are not 
 |      supported.
 |      write dicts data into a json file
 |      input: data, dicts type
 |             filename, string type
 |  
 |  writeJson(self, data, filename)
 |      write dicts data into a json file
 |      input: data, dicts type
 |             filename, string type
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from mytools.MyCommon.MyCommon:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)

Help on class MyCommon in mytools:

mytools.MyCommon = class MyCommon(builtins.object)
 |  docstring for MyCommon
 |  In this module, I pack some frequenty used functions
 |  
 |  Methods defined here:
 |  
 |  __init__(self)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  getCommon(self, dict1, dict2)
 |      get the common key of the two dicts
 |      and get a dicts with the difference value
 |      as a new value.
 |  
 |  loadFile(self, filename)
 |      load data from the file
 |      If it is a yml or yaml type,
 |      yaml module would be used,
 |      if it is a json type, json module
 |      would be used, any other formats are not 
 |      supported
 |      input: filename, string type
 |      return: data, dicts type
 |  
 |  loadJson(self, filename)
 |      load data from the json file
 |      input: filename, string type
 |      return: data, dicts type
 |  
 |  setDirs(self, dirs)
 |      setup for the data directory
 |  
 |  setFileDirs(self, filename)
 |      setup for the filename with absolute path
 |      input: filename, such as "/home/test/data.txt"
 |  
 |  setFilename(self, filename)
 |      setup for the filename with relative path
 |      input: filename, such as "data.txt"
 |  
 |  writeCSV(self, table_array, filename)
 |      把数组写成csv文件输出，以tab符作为间隔符号
 |      同时，应去除原始文本中多余的tab符
 |      
 |      write the array into a csv file with the tab as 
 |      delimiters. On the same time, extraordinary tabs 
 |      are deliminated.
 |  
 |  writeFile(self, data, filename)
 |      write data to the file.
 |      If it is a yml or yaml type,
 |      yaml module would be used,
 |      if it is a json type, json module
 |      would be used, any other formats are not 
 |      supported.
 |      write dicts data into a json file
 |      input: data, dicts type
 |             filename, string type
 |  
 |  writeJson(self, data, filename)
 |      write dicts data into a json file
 |      input: data, dicts type
 |             filename, string type
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)

Help on class NameAll in mytools:

mytools.NameAll = class NameAll(builtins.object)
 |  docstring for Rename
 |  rename the files in the current directory
 |  
 |  Methods defined here:
 |  
 |  __init__(self)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  arr2string(self, arr)
 |      array to string
 |      ["a",'b','c'] -> 'abc'
 |      one line is enough 
 |      string = "".join(arr)
 |  
 |  getCapitalize(self, filename)
 |      get the capitalized format of a string array
 |      map function is used here
 |      filename = list(map(self.normallize,filename))
 |  
 |  getCurrentFiles(self)
 |      get the filenames in the current directory,
 |      with the help of ls command
 |  
 |  getNewFilename(self, filename)
 |      get the new filename of a old one,
 |      characters like [?()[]'"{}#&/\,. would
 |      be deliminated
 |      "a .. ? b ..pdf" -> "AB.pdf"
 |  
 |  getStdStr(self, filename)
 |      get a standard string
 |      "ab ac ab" -> "AbAcAb"
 |      input: filename, string type
 |      return: a new string
 |  
 |  getSuffixIndex(self, filename)
 |      get the suffix of a string
 |      such as : foo.xxx -> -3
 |      filename[-3:] = "xxx"
 |  
 |  normallize(self, name)
 |      get the capitalized format of a word
 |  
 |  run(self)
 |      rename the files with the help of mv 
 |      command after we get the new names
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)

Help on class Triangle in mytools:

mytools.Triangle = class Triangle(builtins.object)
 |  docstring for Triangle
 |  There are a set of tools to deal with triangles, to get
 |  the all kinds of properties of a triangle, you can get 
 |  area, perimeter, radius or center of the inscribed circle,
 |  circumscribed circle or escribed circles. You can get 
 |  lengths, angles, cosine values of the angles. Also, you can
 |  get the equations of orthogonal lines, middle lines, middle
 |  orthogonal lines or angular bisectors.
 |  
 |  Methods defined here:
 |  
 |  __init__(self)
 |      In this module, a point is denoted by [a,b],
 |      and a line is denoted by [A,B,C]
 |      Parameters used in this module are listed here:
 |      self.vertices:
 |          three vertices of the triangle
 |      self.lengths:
 |          lengths of the three sides of the triangle
 |      self.angles:
 |          three angles (0-180) of the triangle
 |      self.cosines:
 |          cosine values of the three angles
 |      self.area:
 |          the area of the triangle
 |      self.sideLines:
 |          three sides of the triangle
 |      self.orthoLines:
 |          three orthogal lines of the triangle
 |      self.midLines:
 |          three middle lines of the triangle
 |      self.midOrthoLines:
 |          three middle orthogonal lines of the triangle
 |      self.bisectLines:
 |          three inner angular bisectors  
 |          and three outer angular bisectors
 |      self.insCenter:
 |          the center of the inscribed circle of the triangle
 |      self.insRadius:
 |          the radius of the inscribed circle of the triangle
 |      self.esCenters:
 |          three centers of the escribed circle of the triangle
 |      self.esRadii:
 |          three radii of the escribed circle of the triangle
 |      self.orthoCenter:
 |          the orthogonal center of the triangle
 |      self.weightCenter:
 |          the weight center of the triangle
 |      self.circumRadius:
 |          the radius of the circumscribed circle
 |      self.circumCenter:
 |          the center of the circumscribed circle
 |      self.orthoPoints:
 |          three orthogonal points of the triangle
 |      self.orders:
 |          indeces of the three vertix-vertix pairs
 |  
 |  draw(self)
 |      draw the triangle, inscribed center
 |      and three escribed centers
 |  
 |  drawLine(self, point1, point2)
 |      draw the line with two points
 |      input: two points
 |  
 |  getAngles(self)
 |      get three angles of the triangle
 |  
 |  getArea(self)
 |      get the area of the triangle
 |  
 |  getBisectLines(self)
 |      get six angular bisectors of the triangleof the triangle
 |  
 |  getCircumCenter(self)
 |      get the center of the circumscribed circle of the triangle,
 |      which is the intersection point of two middle orthogonal
 |      lines
 |  
 |  getCircumRadius(self)
 |      get the radius of the circumscribed cricle of the triangle
 |  
 |  getCosineByIndex(self, i, j)
 |      get cosine values give the indeces of vertices
 |      input: indeces i and j, the third one should be
 |              3 - i - j
 |      return: the cosine value of the angle
 |  
 |  getCosines(self)
 |      get three cosine values of the angles
 |  
 |  getDist(self, vertix1, vertix2)
 |      get the L2 distance of two vertices
 |  
 |  getEsCenters(self)
 |      get three centers of the escribed circle of the triangle
 |      s = (aA+bB+cC)/2
 |      p = (a+b+c)/2
 |      p_rA = (s - aA)/(p - a)
 |      p_rB = (s - bB)/(p - b)
 |      p_rC = (s - cC)/(p - c)
 |  
 |  getEsRadii(self)
 |      get three radii of the escribed circle of the triangle
 |  
 |  getInsCenter(self)
 |      get the center of the inscribed circle of the triangle
 |      s = (aA+bB+cC)/2
 |      p = (a+b+c)/2
 |      p_r = s/p
 |  
 |  getInsRadius(self)
 |      get the radius of the inscribed circle of the triangle
 |  
 |  getInterVertix(self, line1, line2)
 |      get the intersection of two lines
 |      input: two lines
 |      return: a point
 |  
 |  getLengths(self)
 |      get three side length of the triangle
 |  
 |  getMidLines(self)
 |      get three middle lines
 |  
 |  getMidOrthoLine(self, vertix1, vertix2)
 |      get a middle orthogonal line
 |      input: two points
 |      return: a line
 |  
 |  getMidOrthoLines(self)
 |      get three middle orthogonal lines
 |  
 |  getMidPoint(self, vertix1, vertix2)
 |      middle point of two points
 |  
 |  getNewOrder(self, array)
 |      get the inverse order of a array
 |      with a length 3
 |  
 |  getOrthoCenter(self)
 |      get the orthogonal center of the triangle
 |  
 |  getOrthoLine(self, vertix1, vertix2, vertix3)
 |      get the orthogonal line give three  vertices
 |      a point vertix3 on the line, 
 |      which is orthogonal to the one through vertix1 and vertix2
 |  
 |  getOrthoLineByIndex(self, i, j, k)
 |      get the orthogonal line give three indeces of the vertices
 |  
 |  getOrthoLines(self)
 |      get three orthogonal lines of the triangle
 |      AA', BB', CC'
 |  
 |  getOrthoPoints(self)
 |      get three orthogonal points of  the triangle
 |  
 |  getPerimeter(self)
 |      get the perimeter of the triangle
 |  
 |  getSideLines(self)
 |      get three side lines
 |  
 |  getVectorAngle(self, vec1, vec2)
 |      input: two vectors
 |      return: cosine value of their angle
 |      \cos    heta = A\dot B/(||A|| * ||B||)
 |  
 |  getVerticeAngle(self, vertex1, vertex2, vertex3)
 |      input: three vertices
 |      return: cosine value of the angle
 |      the second vertix is the center of the angle
 |  
 |  getVerticesLine(self, vertix1, vertix2)
 |      (x1,y1), (x2,y2) 
 |      (y - y1)/(y2 - y1) = (x - x1)/(x2 - x1)
 |      (y2 - y1)(x - x1) + (x1 - x2)(y - y1)
 |      input: two vertices
 |      return: a line [A,B,C]
 |  
 |  getVertixSlope(self, vertix1, slope)
 |      get the line with a vertix and slope
 |      input: vertix [x,y] and a slope [A,B]
 |      return: a line [A,B,C]
 |  
 |  getWeightPoint(self)
 |      weight point of the triangle
 |  
 |  isOrtho(self, line1, line2)
 |      input: two lines
 |      return: bool, if two lines are orthogonal
 |  
 |  isTriangle(self)
 |      judge if it is a triangle
 |      a + b > c -> (a+b+c)/2 > c
 |  
 |  printArray(self, prefix, labels, array)
 |      input: prefix, such as "length"
 |      input: labels, such as ["A"]
 |      array: array type, such as [1] or [[1,2]]
 |  
 |  printLengths(self)
 |      print out three lengths of the triangle
 |  
 |  printTriInfo(self)
 |      print out the properties of the triangle
 |  
 |  printVertices(self)
 |      print out three vertices of the triangle
 |  
 |  run(self)
 |      get the properties of the triangle 
 |      and print them out
 |  
 |  setLengths(self, a, b, c)
 |      set the lengths given three values
 |  
 |  setVertices(self, vertices)
 |      setup for the coordinate of vertices
 |  
 |  test(self)
 |      test the module 
 |      and draw the triangle, inscribed center
 |      and three escribed centers
 |  
 |  testBisect(self)
 |      test the bisection lines.
 |      judge if the inner bisection lines 
 |      are orthogonal to the outer ones.
 |      There are three orthogonal pairs.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)

Help on class TurtlePlay in mytools:

mytools.TurtlePlay = class TurtlePlay(builtins.object)
 |  docstring for TurtlePlay
 |  It is module for presentation of the usage of turtle module
 |  
 |  Methods defined here:
 |  
 |  __init__(self)
 |      self.theta: 
 |          rotation angle for each time
 |      self.speed:
 |          frames per second
 |          such as : self.speed = 4
 |          the panel will be drawed 4 times in a second
 |      self.length:
 |          length of the line
 |      self.number:
 |          cycles number 
 |      
 |      All the above parameters were initialized to be None
 |  
 |  draw(self)
 |      draw the pattern
 |      length is smaller and smaller in each step
 |  
 |  drawPolygons(self, n)
 |      draw polygons 
 |      add length by 3.5 after drawing a polygon
 |  
 |  drawSquares(self)
 |      draw squares
 |      add length by 3.5 after drawing a square
 |  
 |  initParas(self)
 |      initialize the parameters
 |  
 |  polygon(self, n)
 |      draw a single polygon
 |      with the shape turtle
 |  
 |  printParas(self)
 |      print out all the parameters
 |      including theta, length, number and speed
 |  
 |  setColor(self, colors)
 |      set the color 
 |      input: colors, array with two items
 |      such as : ["red","red"]
 |  
 |  setLength(self, length)
 |      set the length of the line
 |  
 |  setNumber(self, number)
 |      set the number of cycles
 |  
 |  setParas(self, theta, length, number, speed)
 |      setup for all the parameters
 |      including theta, length, number and speed
 |  
 |  setSpeed(self, speed)
 |      set the fps(frames per second)
 |  
 |  setTheta(self, theta)
 |      set the rotation angle
 |  
 |  square(self)
 |      draw a single square
 |      with the shape turtle
 |  
 |  test(self)
 |      test for drawing the squares
 |  
 |  testPolygon(self)
 |      test for drawing the polygons
 |      n was set to be 5
 |  
 |  writeImage(self)
 |      write the image into ps and pdf
 |      ps2pdf is a linux command which 
 |      converts a .ps file into a .pdf one
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)

