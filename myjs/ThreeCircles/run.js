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

var points = [];
var center = [width/2,height/2];
var x,y,phi;
var radius  =  center[1] - 50;
// 1-3th point
var scale = 40;
var a = 4,b = 3;
a = a*scale;
b = b*scale;
var c = Math.sqrt(a*a+b*b);
var p = (a+b+c)/2;

points.push([center[0] - a/2,center[1] + b/2]);
points.push([points[0][0]+a,points[0][1]]);
points.push([points[0][0],points[0][1]-b]);


var tags = ["A","B","C","D","E","F","G",
            "H","I","J","K"];

// get all the lines
// lines: AB,BC,CD,DA,BC,DG,DE,DF,EG
var lines = [];
var indeces = [[0,1],[1,2],[0,2]];
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
for(var i = 4; i < 8; i ++)
{
    dashes[i] = [1,2];
    colors[i] = "orange";
}
for(var i = 8; i < 12; i ++)
{
    dashes[i] = [2,3];
    colors[i] = "green";
}

var circles = [];
var radii = [p-c,p-b,p-a];
for(var i = 0; i < radii.length; i ++)
{
    circles.push([points[i][0],points[i][1],radii[i]]);
}



var circle = svg.selectAll("circle")
                .data(circles)
                .enter()
                .append("circle")
                .attr("cx",function (d,i) {
                  return circles[i][0];
                })
                .attr("cy",function (d,i) {
                  return circles[i][1];
                })
                .attr("r",function (d,i) {
                  return circles[i][2];
                })
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
