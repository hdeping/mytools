/*

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-03-18 13:40:25
    @project      : 0.1
    @version      : learning d3
    @source file  : run.js

============================

*/

    
var width  = 1200;
var height = 800;

// var width  = 1920;
// var height = 1080;

var background_color = "white";
var line_color = "red";


var svg = d3.select("#tree").append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("fill",background_color)
    .append("g")
    .attr("transform", "translate(0,0)");


// points A,B,C,D,E,F,G
// 

var theta = [-90,175,30,65];
var points = [];
var center = [width/2,height/2];
var x,y,phi;
var radius  =  center[1] - 50;
for(var i = 0; i < 4; i ++)
{ 
    phi = theta[i]*Math.PI/180;
    x = radius*Math.cos(phi) + center[0];
    y = radius*Math.sin(phi) + center[1];
    points.push([x,y]);
}
points[3][0] += 0;
// get E,F and G 4-th ~ 6-th
points.push(getOrthoPoint(3,0,1));
points.push(getOrthoPoint(3,1,2));
points.push(getOrthoPoint(3,0,2));
// get three ortho-points and the orthocenter
// 7-th ~ 10-th
points.push(getOrthoCenter(0,1,2));
points.push(getOrthoPoint(0,1,2));
points.push(getOrthoPoint(1,2,0));
points.push(getOrthoPoint(2,0,1));
var tags = ["A","B","C","D","E","F","G",
            "H","I","J","K"];

console.log("line equation");

console.log(getLineEqn(points[0],points[1]));
console.log(points);

// get all the lines
// lines: AB,BC,CD,DA,BC,DG,DE,DF,EG
var lines = [];
var indeces = [[0,1],[1,2],[0,2],
              [1,3],[2,3],[3,4],
              [3,5],[3,6],[4,5],
              [5,6],[0,4],[2,6],
              [3,7],[0,8],[1,9],
              [2,10]];
var dashes = [];

var colors = [];
for(var i = 0; i < indeces.length; i ++)
{
    var tmp = [];
    tmp.push(points[indeces[i][0]]);
    tmp.push(points[indeces[i][1]]);
    lines.push(tmp);
    dashes.push([]);
    colors.push(line_color);
}
for(var i = 5; i < 10; i ++)
{
    dashes[i] = [1,2];
    colors[i] = "orange";
}
for(var i = 13; i < 16; i ++)
{
    dashes[i] = [2,1];
    colors[i] = "blue";
}

var circle1 = [1];
var circle = svg.selectAll("circle")
                .data(circle1)
                .enter()
                .append("circle")
                .attr("cx",center[0])
                .attr("cy",center[1])
                .attr("r",radius)
                .attr("stroke-width",2)
                .attr("stroke","blue");

var sides = svg.selectAll("line")
               .data(lines)
               .enter()
               .append("line")
               .attr("x1",function (d,i) {
                 return lines[i][0][0];
               })
               .attr('y1', function (d,i) {
                 return lines[i][0][1];
               })
               .attr('x2', function (d,i) {
                 return lines[i][1][0];
               })
               .attr('y2', function (d,i) {
                 return lines[i][1][1];
               })
               .attr("stroke",function (d,i) {
                 return colors[i];
               })
               .attr("stroke-width",2)
               .attr("stroke-dasharray",function (d,i) {
                 return dashes[i];
               });

// add texts
var dx = 10,dy = 10;
var texts = svg.selectAll("text")
               .data(points)
               .enter()
               .append("text")
               .attr("x",function (d,i) {
                 return points[i][0] + dx;
               })
               .attr('y', function (d,i) {
                 return points[i][1] + dy;
               })
               .text(function (d,i) {
                 return tags[i];
               })
               .attr("fill","green");

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




