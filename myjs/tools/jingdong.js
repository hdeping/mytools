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


String.prototype.trim = function(input){
    var output = "";
    for(var i = 0; i < this.length; i ++)
    {
        if (this[i] != input) {
            output += this[i];
        }
    }
    
    return output;
    
};
String.prototype.toArray = function () {
    var res = [];
    output = this.split('\n');
    for(var i = 0; i < output.length; i ++)
    {
        if (output[i] != "") {
            res.push(output[i]);
        }
    }
    return res;
};

var title = document.querySelector(".time-txt");
title = title.textContent + ".json";

var tables = document.querySelectorAll("tbody");
var contents = [];
var line;
var results = [];
for(var i = 0; i < tables.length; i ++)
{
    line = tables[i].textContent.trim(' ');
    line = line.trim("\t");
    line = line.toArray(); 
    results.push(line);
    console.log(line);
}
exportRaw(title, results);

