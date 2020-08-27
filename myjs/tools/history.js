javascript:
function sleep (time) {
  return new Promise((resolve) => setTimeout(resolve, time));
};
function fakeClick(obj) { 
    var ev = document.createEvent("MouseEvents");
    ev.initMouseEvent("click", true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
    obj.dispatchEvent(ev);
};

function exportRaw(name, data) {
    var urlObject = window.URL || window.webkitURL || window;
    var export_blob = new Blob([data]);
    var save_link = document.createElementNS("http://www.w3.org/1999/xhtml", "a");
    save_link.href = urlObject.createObjectURL(export_blob);
    save_link.download = name;
    fakeClick(save_link);
};

var nodes = document.querySelectorAll(".tree-title");
var files;
var filename = prompt("请输入文件名","ningde.json,waiting_time,begin_page");
var waiting_time,begin_page;
filename = filename.split(",");
if (filename.length == 1) {
    waiting_time = 100;
    begin_page = 0;
}
else {
    waiting_time = parseInt(filename[1]);
    if (filename.length == 2) {
        begin_page = 0;
    }
    else {
        begin_page = parseInt(filename[2]);
    }
    
}
filename = filename[0];
var total = [];
var num = nodes.length;
(async function() {
    for(var i = begin_page; i < num; i ++)
    {   
        nodes[i].click();
        await sleep(waiting_time);  
        files = document.querySelector("iframe");
        files = files.src;
        console.log(i+1,files);
        total.push(files);
    }
    exportRaw(filename, total);
})();

