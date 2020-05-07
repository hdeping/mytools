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

var theta = [-90,175,30,115];
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

