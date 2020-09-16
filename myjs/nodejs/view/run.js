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

// ajax here
var xmlhttp;
if (window.XMLHttpRequest)
{
    //  IE7+, Firefox, Chrome, Opera, Safari 浏览器执行代码
    xmlhttp=new XMLHttpRequest();
}
else
{
    // IE6, IE5 浏览器执行代码
    xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
}



get_job_running();

var count_refresh = 0;

function refresh () {
    count_refresh ++;
    
    window.onload=function(){
        document.body.innerHTML = "";

    }
   

    xmlhttp.open("POST",
        "http://210.45.125.225:3000/refresh",
        true);
    xmlhttp.send();
    console.log(xmlhttp.readyState);
    

    // if (xmlhttp.readyState==4 && xmlhttp.status==200)
    if (xmlhttp.readyState==4 )
    {
        console.log("refreshing",count_refresh);
        get_job_running();
    }   
    
}



function get_job_running(){

    xmlhttp.open("GET","jobs.json",true);
    xmlhttp.send();

    xmlhttp.onreadystatechange=function()
    {  
        //console.log(xmlhttp.readyState,xmlhttp.responseText);
        if (xmlhttp.readyState==4 && xmlhttp.status==200)
        {
            var data  = xmlhttp.responseText;
            // console.log(data);
            
            data = JSON.parse(data);
            let count = 0;
            var running;
            for(var key in data){
                count ++;
                if (count == 1) {
                    running = data[key];
                    get_all(key,running);
                }
                else if (count == 2) {
                    running = data[key];
                    get_100(key,running);
                }
                else if (count == 3) {
                    // console.log("I am .17");
                    
                    running = data[key];
                    get_17(key,running);
                }
                else if (count == 4) {
                    // console.log("I am .17");
                    
                    running = data[key];
                    get_7(key,running);
                }
                // console.log(count,key,data[key]);
                
            }  
            // draw all the picture
            // console.log(running);
        }
    };

}