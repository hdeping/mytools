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


function get_17(key,running){
    var block = document.createElement("div");
        block.className = "block";
    block.appendChild(add_title(key,2));
    
    
    // var running = [];
    
    
    
    svg_url = "http://www.w3.org/2000/svg";


    // for .17 server 
    // node 1 to 10
              
        
    for(var i = 1; i < 11;  i ++)
    {
        div = add_node17(i,running);
        block.appendChild(div); 
    }
    
    document.body.appendChild(block);
    draw17(running);
}
function add_title(title,id) {
    // add a div
    var div = document.createElement("div");
    div.className = "title"+id;
    var text =document.createTextNode(title);
    div.appendChild(text);
    // div.appendChild(nodename);
    return div;
    
}


// index:1-14
function add_node17 (index,running) {
    // add a div
    var div = document.createElement("div");
    div.className = "node17";
    div.id = "node" + index;
    // add child div
    var nodename = document.createElement("div");
    nodename.className = "nodename";
    var node_name;
    if (index < 10) {
       node_name = "0"+ index;
    }
    else{
        node_name = ""+index;
    }
    var text =document.createTextNode(node_name);
    nodename.appendChild(text);
    div.appendChild(nodename);

    

    // var text =document.createTextNode("节点01");
    // svg
    var svgObject = document.createElementNS(svg_url,"svg");
    svgObject.setAttribute("id", "svg1"+index);

    // draw(running);
    div.appendChild(svgObject);
    return div;
    
}


function draw17(running) {
    
 
    for(var i = 1; i < 11; i ++)
    {
        const svg = d3.select("#svg1"+i);

        var jobs = running["node"+i];

        draw_svg17(svg,jobs);
    }
    
}

function draw_svg17 (svg,running) {
    var side = 40;
    // console.log(".17",running);
    
    var num =  running[0] + running[1];
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
        if (count > running[0] ) {
            color = 'green';
        }
        else{
            color = 'red';
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