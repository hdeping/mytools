/*

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-05-31 10:31:19
    @project      : HPC monitor
    @version      : 1.0
    @source file  : server7.js

============================

*/


function get_all(key,running){
    var block = document.createElement("div");
        block.className = "block";
        key += " \nrunning " + sum(running);
        key += " \nremain " + (392 - sum(running));
        key += " \nwaiting " + running[14];
        block.appendChild(add_title(key,1));
    
    
    // var running = [];
    
    
    
    svg_url = "http://www.w3.org/2000/svg";
        
    for(var i = 1; i < 15;  i ++)
    {
        // console.log("225",i,running[i-1]);
        
        div = add_node(i,running);
        block.appendChild(div); 
    }
    
    document.body.appendChild(block);
    draw(running);
}
function sum(running){
    var res = 0;
    // sum of 1 to 14
    for(var i = 0; i < 14;  i ++)
    {
        res += running[i];
    }
    return res;
    
}

// index:1-14
function add_node (index,running) {
    // add a div
    var div = document.createElement("div");
    div.className = "node";
    div.id = "node" + index;
    // add child div
    var nodename = document.createElement("div");
    nodename.className = "nodename";
    if (index < 10) {
        nodename_string = "0" + index;
    }
    else{
        nodename_string = "" + index;
    }
    var text =document.createTextNode(nodename_string);
    nodename.appendChild(text);
    div.appendChild(nodename);

    

    // var text =document.createTextNode("节点01");
    // svg
    var svgObject = document.createElementNS(svg_url,"svg");
    svgObject.setAttribute("id", "svg0"+index);

    // draw(running);
    div.appendChild(svgObject);
    return div;
    
}


function draw(running) {
    
 
    for(var i = 1; i < 15; i ++)
    {
        const svg = d3.select("#svg0"+i);

        draw_svg(svg,running,i-1);
    }
    
}

function draw_svg (svg,running,index) {
    var side = 40;
    var num =  28;
    var count = 0;
    
    var interval = 2;
    var width  = side*num;
    var height = side;
    // clear the contents
    // svg.selectAll("svg > *").remove(); 
    svg.attr("width",width)
       .attr('height', height);
    
    
    for(var i = 0; i < num; i++){
        
        count ++;
        if (count > running[index] ) {
            color = 'green';
        }
        else{
            color = 'red';
        }
        if(running[15+index] == "down"){
            color = "grey";
        }
        
        var y = interval;
        var x = i*side + interval;
        svg.append('rect')
            .attr('x', x)
            .attr('y', y)
            .attr('width', side - 2*interval)
            .attr('height', side - 2*interval)
            .style('fill', color);
        
    }
        
}