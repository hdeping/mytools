/*

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-05-05 02:10:49
    @project      : tree artist
    @version      : 0.1
    @source file  : run.js

============================

--------------------- 
作者：风火一回 
来源：CSDN 
原文：https://blog.csdn.net/mafan121/article/details/50435530 
版权声明：本文为博主原创文章，转载请附上博文链接！

*/

 

//定义布局范围
var width  = 900;
var height = 600;
var treeWidth = width-100;
var treeHeight = height-100;
//绘制连线
var rect_height  = 50;
var rect_width   = 120;
// length of the string
// 1,2,3,4
var text_x_offset = [5,22,9,7];
var text_y_offset = 37.5;
var link_x_offset = [5,-20 ,9,7];
var link_y_offset = 47.5;
// round corner
var round = 10;



var hierarchy = d3.hierarchy(data);
// var tree = d3.tree();
//定义D3树布局范围
var tree = d3.tree()
        // size 
        .size([treeWidth,treeHeight])
        .separation(function(a, b) { 
           //设置相隔节点的间距，a、b节点相隔
           // if (a.parent == b.parent){
           //    return 2;
           // }
           // else{
           //    return 1;
           // }
           return 1;
            
        });
var links = tree(hierarchy).links();
var all_nodes = [];
get_all_nodes(hierarchy);
// print(all_nodes)
console.log(all_nodes);




    
//绘制svg图形
var svg = d3.select("#relevanceRuleConfig").append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", "translate(40,40)");//定义偏移量
 
 
//加载数据

// var nodes = tree.nodes(data);   //获取所有节点信息
    


var link = svg.selectAll(".link")
          .data(links)
          .enter()
          .append("path")
          .attr("class", "link")
          // .attr("d", diagonal);
          .attr("d", function(d,i){
            
            
             var res = get_link(d.source, d.target);
             // console.log("links",i,res);
             return res;
          });
    
//绘制节点
console.log("get nodes");

var node = svg.selectAll(".node")
          .data(all_nodes)
          .enter()
          .append("g")
          .attr("class", "node")
          .attr("transform", function(d) { 
            // console.log(d.x,d.y);

            
            return "translate(" + (d.x- rect_width/2) + "," + d.y + ")"; 
          });
    
//添加节点图标
node.append("rect")
    .attr("height", rect_height)
    .attr("width", rect_width)
    .attr("rx",function(d,i){
      if (d.name == "好瓜" || 
          d.name == "坏瓜") {
        console.log(i,d.name);
      
       return rect_width/2;
      }
      else{
        return round;
      }
    })
    .attr("fill",function(d,i){
      if (d.name == "好瓜"){
       return "green";
      }
      else if (d.name == "坏瓜"){
       return "red";
      }
      else{
        return "white";
      }
    })

    .attr("ry",function(d,i){
      if (d.name == "好瓜" || 
          d.name == "坏瓜") {
        return rect_height/2;
      }
      else{
        return round;
      }
    });


    
//添加节点显示文本
node.append("text")
    //定义文本显示x轴偏移量
    // .attr("dx", function(d) { 
    //     // return d.children ? -8 : 8; 
    //     return -8;
    // })
    //定义文本显示x轴偏移量
    .attr("class","text")
    .attr("dx", function(d,i){
      var num = d.name.length - 1;
      return text_x_offset[num];
    })
    //定义文本显示y轴偏移量
    .attr("dy", text_y_offset)
    .style("text-anchor", function(d) {
        //文字对齐显示 
        return d.children ? "end" : "start"; 
    })
    .attr("fill",function(d,i){
      if (d.name == "好瓜"){
       return "white";
      }
      else if (d.name == "坏瓜"){
       return "black";
      }
      else{
        return "blue";
      }
    })
    .text(function(d) {
        return d.name; 
    });


console.log(links);

//添加链接显示文本
d3.select("svg")
    .selectAll(".link_text")
    .data(links)
    .enter()
    .append("text")
    .attr("class","link_text")
    .attr("dx", function(d,i){
      var num = d.target.data.link.length - 1;
      return link_x_offset[num];
    })
    //定义文本显示y轴偏移量
    .attr("dy", link_y_offset)
    .attr("x", function(d,i){
       var x = (d.source.x + d.target.x)/2;
       return x;
    })
    .attr("y", function(d,i){
       var y = (d.source.y + d.target.y + rect_height)/2;
       return y;
    })
    .style("text-anchor", function(d) {
        //文字对齐显示 
        return d.children ? "end" : "start"; 
    })
    .text(function(d) {
        return d.target.data.link; 
    });


function get_link (source,target) {


    var x1 = source.x;
    var y1 = source.y + rect_height;
    var x2 = target.x;
    var y2 = target.y ;

    // var x1 = source.y;
    // var y1 = source.x;
    // var x2 = target.y;
    // var y2 = target.x;

    var center = (x1 + x2) / 2;
    
    res = "M " ; 
    res += x1;
    res += " "  ;
    res += y1;
    res += "C"  ;
    res += center ;
    res += ","  ;
    res += y1  ;
    res += " "  ;
    res += center ;
    res += ","  ;
    res += y2;
    res += " "  ;
    res += x2;
    res += ","  ;
    res += y2;

    // res +=  " L ";
    // res += target.y  ;
    // res += " "  ;
    // res += target.x;

    return res;
}

function get_all_nodes(node){
    var info = {name:node.data.name,
                   x:node.x,
                   y:node.y};
 
    all_nodes.push(info);
    if (node.children){
      var length = node.children.length;
      for(var i = 0; i < length; i ++)
      {
          get_all_nodes(node.children[i]);
      }
      
    }

}