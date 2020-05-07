/*

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-05-07 23:00:44
    @project      : triangles in js
    @version      : 1.0
    @source file  : triangles.js

============================

*/

function getLineEqn (point1,point2) {
  // (x - x1)(y2-y1) - (x2-x1)(y - y1) = 0 
  var line = [];
  var dx,dy;
  dx = point2[0] - point1[0];
  dy = point2[1] - point1[1];

  line.push(dy);
  line.push(-dx);
  var C = point1[1]*dx - point1[0]*dy;
  line.push(C);

  return line; 
}

function getOrthoPoint(i,j,k) {

    var point1 = points[i];
    var point2 = points[j];
    var point3 = points[k];

    var p = getOrthoPointByPoints(point1,point2,point3);
    return p;

} 

function getOrthoPointByPoints (point1,point2,point3) {
    // perpendicular point of point1 on the line point2-point3

    var line = getLineEqn(point2,point3);
    var A = line[0];
    var B = line[1];
    var C = line[2];
    var x0 = point1[0], y0 = point1[1];
    var g = (A*x0 + B*y0 + C)/(A*A+B*B);
    var p = [x0 - A*g,y0 - B*g];

    return p;

}   
function getDet2 (a,b,c,d) {
  // get the determinant of 2*2 matrix
  // |a b|
  // |c d|
  return a*d-b*c;
} 
function getVecDet2 (v1,v2) {
  // get the determinant of 2*2 matrix
  // |a b|
  // |c d|
  return getDet2(v1[0],v1[1],v2[0],v2[1]);
} 
function getLineInter (line1,line2) {
  // get the intersection point of the two lines 
  var A = getDet2(line1[1],line1[2],line2[1],line2[2]);
  var B = -getDet2(line1[0],line1[2],line2[0],line2[2]);
  var C = getDet2(line1[0],line1[1],line2[0],line2[1]);

  var p = [A/C,B/C];
  return p;
}     

function getOrthoLineByPoints(p1,p2,p3) {
  var line1 = getLineEqn(p2,p3);
  var A = line1[0];
  var B = line1[1];
  var C = line1[2];

  var res = [B,-A];
  res.push(A*p1[1]-B*p1[0]);

  return res;
}

function getOrthoCenterByPoints(p1,p2,p3) {
  var line1 = getOrthoLineByPoints(p1,p2,p3);
  var line2 = getOrthoLineByPoints(p2,p1,p3);
  return getLineInter(line1,line2);
}

function getOrthoCenter(i,j,k) {

  var p1 = points[i];
  var p2 = points[j];
  var p3 = points[k];

  var line1 = getOrthoLineByPoints(p1,p2,p3);
  var line2 = getOrthoLineByPoints(p2,p1,p3);
  return getLineInter(line1,line2);

}
function getBisectionPoint (p1,p2,p3,scale) {
  // get a point on the bisection line 
  // scale is the length scale

  var line1 = getNormVector(p2,p1);
  var line2 = getNormVector(p2,p3);
  var bisection = [];
  var x;
  for(var i = 0; i < line1.length; i ++)
  {
      x  = scale*(line1[i]+line2[i])/2;
      x += p2[i];
      bisection.push(x);
  }


  return bisection;

}

function getBisectionLine (p1,p2,p3) {
    var p = getBisectionPoint(p1,p2,p3,1);
    return getLineEqn(p2,p);
}

function getVector (p1,p2) {
  // vector from p1 to p2 

  var res = [];
  for(var i = 0; i < p1.length; i ++)
  {
      res.push(p2[i] - p1[i]);
  }
  return res;

}

function getNormVector (p1,p2) {
  // vector from p1 to p2 
  
  var res = getVector(p1,p2);
  var length = getVecLength(res);

  for(var i = 0; i < res.length; i ++)
  {
      res[i] = res[i]/length;
  }

  return res;

}

function getVecLength (res) {
  var length = 0;
  for(var i = 0; i < res.length; i ++)
  {
      length += Math.pow(res[i],2);
  }

  length = Math.sqrt(length);
  return length;
}

function getDist (p1,p2) {
  // get the distance between two points
  var vec = getVector(p1,p2);
  return getVecLength(vec);
}

function getQuadraticRoots (a,b,c) {
    var delta = Math.sqrt(b*b-4*a*c);
    var res = [];
    res.push((-b+delta)/(2*a));
    res.push((-b-delta)/(2*a));
    return res;
}

function getTriArea (p1,p2,p3) {

  var v1 = getVector(p1,p2);
  var v2 = getVector(p1,p3);
  var area = getVecDet2(v1,v2);
  return Math.abs(area/2);

}
function getQuadArea (p) {
    // p: four points
    var a1 = getTriArea(p[0],p[1],p[2]);
    var a2 = getTriArea(p[0],p[2],p[3]);
    return a1+a2;
}
function getQuadCircle (p) {
  // p: four points 
  // r = 2S/(a+b+c+d)
  var a = getDist(p[0],p[1]);
  var c = getDist(p[2],p[3]);
  var area = getQuadArea (p);
  var radius = area/(a+c);

  // intersection of two bisections
  var line1 = getBisectionLine(p[0],p[1],p[2]);
  var line2 = getBisectionLine(p[1],p[2],p[3]);
  var p1 = getLineInter(line1,line2);
  return [p1[0],p1[1],radius];
}