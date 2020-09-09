// 图片统计
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
var images = document.querySelectorAll(".imagewrap-inner>a");
var n = images.length;
var links = [];
for(var i = 0; i < n; i ++)
{
    links.push(images[i].src);
}