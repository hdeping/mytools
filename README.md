# Introduction to mytools module
my own tools for gui, math calculations and so on. If you have a linux or Macos environment, you should move the mytools
directory to a directory within PYTHONPATH(you can edit it in a .bashrc
or .zshrc file). If you have a windows environment, you can edit PYTHONPATH
following the guide as it shows in the link: 
https://blog.csdn.net/Tona_ZM/article/details/79463284

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

![polygon5](/Users/huangdeping/c/02_python/44_mytools/figures/polygon5.png)

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

