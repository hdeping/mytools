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


function get_100(key,running){
    var block = document.createElement("div");
        block.className = "block";
    key += "\nWaiting: "+ running["remain"];
    block.appendChild(add_title(key,1));
    
    
    // var running = [];
    
    
    
    svg_url = "http://www.w3.org/2000/svg";


    let count = 0;      
    for(var key in running)
    {
        count ++;
        if (key == "remain") {
            continue;
        } 
    
        div = add_node100(count,key);
        block.appendChild(div); 
    }
    
    document.body.appendChild(block);
    draw100(running);
}

// index:1-14
function add_node100(index,key) {
    // add a div
    var div = document.createElement("div");
    div.className = "node100";
    div.id = "node" + index;
    // add child div
    var nodename = document.createElement("div");
    nodename.className = "nodename1";
    var text =document.createTextNode(key);
    nodename.appendChild(text);
    div.appendChild(nodename);

    

    // var text =document.createTextNode("节点01");
    // svg
    var svgObject = document.createElementNS(svg_url,"svg");
    svgObject.setAttribute("id", "svg2"+index);

    // draw(running);
    div.appendChild(svgObject);
    return div;
    
}


function draw100(running) {
    
    var count = 0;
    for(var key in running)
    {
        count ++;
        if (key == "remain") {
            continue;
        }
        const svg = d3.select("#svg2"+count);

        var jobs = running[key];
        // console.log(key,jobs);

        draw_svg100(svg,jobs);
    }
    
}

function draw_svg100(svg,running) {
    var side = 40;
    var num =  running[0] + running[1];
    
    
    var count = 0;
    
    var interval = 2;
    var width  = side*num;
    var height = side;
    // console.log("gpu",running,num,width,height);
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