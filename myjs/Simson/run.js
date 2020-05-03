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

    
var width  = 900;
var height = 600;

// var width  = 1920;
// var height = 1080;



var rect_width  = 120;
var rect_height = 120;
var padding     = 10;
var radius      = 60;
var x_begin = 120;
var y_begin = 120;
var background_color = "white";


var svg = d3.select("#tree").append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("fill",background_color)
    .append("g")
    .attr("transform", "translate(0,0)");


// points A,B,C,D,E,F,G

var theta = [90,-135,-60,-80];
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
console.log(points);
console.log(getLineEqn(points[0],points[1]));


// get all the lines
// lines: AB,BC,CD,DA,BC,DG,DE,DF,EG
var lines = [];
var indeces = [[0,1],[1,2],[0,2],
              [1,3],[2,3]];
for(var i = 0; i < indeces.length; i ++)
{
    var tmp = [];
    tmp.push(points[indeces[0]]);
    tmp.push(points[indeces[1]]);
    lines.push(tmp);
}
console.log(lines);




var circle = svg.selectAll("circle")
                .data(lines)
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
               .attr("stroke",2)
               .attr("stroke-width",2);


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
               


