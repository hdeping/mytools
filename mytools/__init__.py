#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-07 00:27:08
    @project      : my python tools
    @version      : 1.0
    @source file  : __init__.py

============================
"""

try:
    from .Draw  import (
        TurtlePlay,
        DrawCurve,
        DrawPig
    )
except Exception:
    print("there is no GUI for .Draw")
try:
    from .utils import (
        NameAll, 
        GetDoc, 
        MyGUI,
        GetLines, 
        OpenFiles,
        RunCommand, 
        MyPdf
    )
except Exception:
    print("there is no GUI for .utils")
try:
    from .Triangle import Triangle
except Exception:
    print("there is no GUI for .Triangle")
try:
    from .ControlADB      import ControlADB
except Exception:
    print("there is no GUI for .ControlADB")
try:
    from .Spider          import Spider
except Exception:
    print("there is no GUI for .Spider")
from .MyCommon        import MyCommon
from .Excel           import Excel
from .DistanceMeasure import DistanceMeasure
