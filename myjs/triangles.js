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