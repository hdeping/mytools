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


function get_7(key,running){
    var block = $("<div></div>").attr("class","block");
    key += "\nWaiting: "+running["remain"];
    block.append(add_title(key,2));
    // var running = [];
    
    svg_url = "http://www.w3.org/2000/svg";


    // for .7 server 
    // node 1 to 12
              
        
    for(var i = 1; i < 13;  i ++)
    {
        div = add_node7(i,running);
        block.appendChild(div); 
    }
    
    $("body").(block);
    draw7(running);
}
function add_title(title,id) {
    var div = $("<div></div>")
              .attr("class","title"+id)
              .text(title);
    return div;
    
}


// index:1-14
function add_node7 (index,running) {
    // add a div
    var div = document.createElement("div");
    div.className = "node7";
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
    svgObject.setAttribute("id", "svg3"+index);

    // draw(running);
    div.appendChild(svgObject);
    return div;
    
}


function draw7(running) {
    
 
    for(var i = 1; i < 13; i ++)
    {
        const svg = d3.select("#svg3"+i);

        var jobs = running["node"+i];

        draw_svg7(svg,jobs);
    }
    
}

function draw_svg7 (svg,running) {
    var side = 40;
    // console.log(".7",running);
    
    var num =  16;
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
        if (running[2] == "-NA-") {
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