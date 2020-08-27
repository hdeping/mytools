// sleep and wait
function sleep (time) {
  return new Promise((resolve) => setTimeout(resolve, time));
};
// save file
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

var total = [],text,playDate;
var count = 0; 
function getReviews (content) {
    var reviews = document.querySelectorAll(content);
    for(var i = 0; i < reviews.length; i ++)
    {
        count += 1;
        text = reviews[i].childNodes[1].title;
        playDate = reviews[i].childNodes[2].textContent;
        playDate = playDate.split("\n");
        playDate[1] = playDate[1].replace(/\s/ig,'');
        playDate.pop();
        console.log(count,text,playDate);
        total.push([count,text,playDate]);
    }
    
};

function getReviews1 (content) {
    var reviews = document.querySelectorAll(content);
    for(var i = 0; i < reviews.length; i ++)
    {
        count += 1;
        text = reviews[i].textContent;
        console.log(text);
        total.push(text);
    }
    
};

function getPages (content,next_page,pages,filename) {

    (async function() {
        getReviews1(content);
        for(var i = 0; i < pages - 1; i ++)
        { 
          document.querySelector(next_page).click();
          await sleep(500); 
          getReviews1(content);
        }
        exportRaw(filename, total);
    })();

};

// var next_page = ".be-pager-next";
// var content = "li[class='small-item fakeDanmu-item']";
// var filename = "xiaowenge.json";
// var pages = document.querySelector(".be-pager-total").textContent;
// pages = parseInt(pages.split(" ")[1]);
var next_page = ".next";
var content = "div>p.text";
var filename = "leijun.json";
var pages = 73;
getPages(content, next_page, pages, filename);