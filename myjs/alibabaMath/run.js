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

    
var width  = 1024;
var height = 768;

// var width  = 1920;
// var height = 1080;



var rect_width  = 120;
var rect_height = 120;
var padding     = 10;
var radius      = 30;
var x_begin = 240;
var y_begin = 240;
var posx = [];
var posy = [];
var x,y;

var row = 3, col = 4;
for(var i = 0; i < col; i ++)
{
    x = x_begin +  i*(rect_width+padding);
    posx.push(x);
}

for(var i = 0; i < row; i ++)
{
    y = y_begin +  i*(rect_height+padding);
    posy.push(y);
}
var dataset = [];
var ii,jj;

var colors = [];
for(var i = 0; i < row*col; i ++)
{
    jj = i%row;
    ii = Math.floor(i/row);
    dataset.push([posx[ii],posy[jj]]);
    colors.push(0);
}



var svg = d3.select("#tree").append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("fill","black")
    .append("g")
    .attr("transform", "translate(0,0)");

var rects = svg.selectAll("rect")
               .data(dataset)
               .enter()
               .append("rect")
               .attr("fill","pink")
               .attr('width', rect_width)
               .attr('height', rect_height)
               .attr('rx', radius)
               .attr('ry', radius)
               .attr("x",function (d,i) {
                 return dataset[i][0];
               })
               .attr("y",function (d,i) {
                 return dataset[i][1];
               })
               .on("click",function (d,i) {
                  if (colors[i] == 0) {
                      d3.select(this).attr("fill","blue");    
                  }
                  else{
                      d3.select(this).attr("fill","pink");     
                  }
                  colors[i] = 1 - colors[i];
                 
               });


