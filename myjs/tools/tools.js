/*

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-08-10 21:59:45
    @project      : some js code collections
    @version      : 1.0
    @source file  : try.js

============================

*/

// sleep and wait
function sleep (time) {
  return new Promise((resolve) => setTimeout(resolve, time));
}
 
(async function() {
  alert('Do some thing, ' + new Date());
  for(var i = 0; i < 5; i ++)
  {
    await sleep(1000);
    alert('当前时间： ' + new Date());   
  }
})();



var jq = "http://libs.baidu.com/jquery/2.0.0/jquery.min.js";
var script = document.createElement('script');
script.type = 'text/javascript';
script.src = jq;
document.body.appendChild(script);


var count = 0;
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

var total = [];
var link,text;
function getNames () {
    var links = $("#impFile").find("a");
    // var links = document.getElementById("impFile").childNodes
    for(var i = 0; i < links.length; i ++)
    {
        count += 1;
        text = links[i].text;
        href = links[i].href;
        console.log(count,text,href);
        total.push([count,text,href]);
    }   
};




(async function() {
    // get the pages
    var pages = $("#page_div").find("b");
    pages = parseInt(pages[0].textContent);
    getNames();
    await sleep(500);
    for(var i = 1; i < pages; i ++)
    {
      $(".nextbtn").click();
      getNames();
      await sleep(500);  
    }
    exportRaw('menghaixian.json', total);
})();


// search the urls
var prefix = "https://www.12309.gov.cn/12309/gj/yn/kms/jnx/zjxflws/index.shtml?channelLevels=";

var lists = $("dl>dt");
console.log(lists.length);

var text,link;
for(var i= 0;i<lists.length;i++){
    link = lists[i].getAttribute("channellevels");
    text = lists[i].childNodes[0].text;
    console.log(i,text,prefix+link);

}


var total = [],text;
function getCodes () {
    var codes = $(".table-responsive").find("td");
    for(var i = 0; i < codes.length; i ++)
    {
        text = codes[i].textContent;
        console.log(text);
        total.push(text);
    }
}

(async function() {
  // get city codes
    getCodes();
    for(var i = 0; i < 81; i ++)
    {
        var a = $(".j-pager");
        a[a.length - 2].click();
        await sleep(800);
        getCodes();
    }
    exportRaw("city_codes.json", total);
})();


// get m3u8
var iframes = document.querySelectorAll("iframe");
var src = "";
for(var i = 0; i < iframes.length; i ++)
{
    src = iframes[i].src;
    var suffix = src.split(".");
    var num = suffix.length;
    suffix = src[num-1];
    if (suffix == "m3u8") {
        break;
    }
}
src = src.split("=");
if (src.length > 1) {
    src = src[1];
}
console.log(src);
window.open(src);



// nature.com
var prefix = "https://www.nature.com";
var urls = $("a[itemprop='url']");
for(var i = 2; i < urls.length; i ++)
{
    console.log(i,urls[i]+"");
    
}



// 淘宝
var total = [],text;
var count = 0; 
function getReviews () {
    var reviews = document.querySelectorAll("div[class='J_KgRate_ReviewContent tb-tbcr-content ']");
    for(var i = 0; i < reviews.length; i ++)
    {
        count += 1;
        text = reviews[i].textContent;
        console.log(count,text);
        total.push(text);
    }
    
}

(async function() {
    getReviews();
  for(var i = 0; i < 4; i ++)
  { 
    document.querySelector(".pg-next").click();
    await sleep(5000); 
    getReviews();
  }
  exportRaw("zhihu.json", total);
})();


// 知乎, 爬取记录


var total = [],text;
var count = 0; 
function getReviews (content) {
    var reviews = document.querySelectorAll(content);
    for(var i = 0; i < reviews.length; i ++)
    {
        count += 1;
        text = reviews[i].textContent;
        console.log(count,text);
        total.push(text);
    }
    
}

function getPages (content,next_page,pages,filename) {

    (async function() {
        getReviews(content);
        for(var i = 0; i < pages - 1; i ++)
        { 
          document.querySelector(next_page).click();
          await sleep(6000); 
          getReviews(content);
        }
        exportRaw(filename, total);
    })();

}

var next_page = "button[class='Button PaginationButton PaginationButton-next Button--plain']";
var content = "div[class='RichText ztext']";
var filename = "zhihu.json";
var pages = 13;
getPages(content, next_page, pages, filename);


// library genesis
var links = document.querySelectorAll("td>a");
var link,url;
var urls = [];
for(var i = 0; i < links.length; i ++)
{
    link = links[i].href;
    url = link.split("/")[3];
    if (url.substring(0,4) == "book") {
        console.log(link);
        urls.push(link);
        window.open(link);
    } 
}

// 中国地理志资料
var nodes = document.querySelectorAll(".tree-title");
var files;
for(var i = 0; i < 10; i ++)
{
    nodes[i].click();
    files = document.querySelector("iframe");
    files = files.contentDocument.body.textContent;
    console.log(files);   
}

// B站加速

javascript:
var dicts = {};
var arr = ["2","1.5","1.25","1","0.75","0.5"];
for(var i = 0;i<arr.length;i++){
    dicts[arr[i]] = i;
}
var speeds = $("ul[class='bilibili-player-video-btn-speed-menu']>li");
var index = prompt("请输入倍数","2,1.5,1.25,1,0.75,0.5等");
index = dicts[index];
speeds[index].click();

// 网络通
// time,0-4,1小时,4小时，11小时，14小时，永久
function sleep (time) {
  return new Promise((resolve) => setTimeout(resolve, time));
}
var exps = document.querySelectorAll("input[name='exp']");
var types = document.querySelectorAll("input[name='type']");
var go = document.querySelector("input[name='go']");

function login (input) {
    if (input.length != 2) {
        alert("two numbers are needed, such as 11,95 and so on");
    }
    else{

        try {
            (async function() {
                type = parseInt(input[0]) - 1;
                exp = parseInt(input[1]) - 1;
                console.log(type,exp);
                types[type].click();
                await sleep(100);
                exps[exp].click();
                await sleep(100);
                go.click();
            })();
        } catch(e) {
            alert(e);
            console.log(e);
        }
    }
}
login(prompt("请输入编号",15));

// get all scripts
var arr = document.querySelectorAll("script")
for(var i = 0, length1 = arr.length; i < length1; i++){
    if (arr[i].src == "") {
        console.log(i+1,arr[i].textContent);
    }
    else {
        console.log(i+1,arr[i].src);   
    }
    
}

// 科大仪器中心
Date.prototype.format = function(fmt) { 
     var o = { 
        "M+" : this.getMonth()+1,                 
        "d+" : this.getDate(),                   
        "h+" : this.getHours(),                  
        "m+" : this.getMinutes(),                
        "s+" : this.getSeconds(),                
        "q+" : Math.floor((this.getMonth()+3)/3),
        "S"  : this.getMilliseconds()             
    }; 
    if(/(y+)/.test(fmt)) {
            fmt=fmt.replace(RegExp.$1, (this.getFullYear()+"").substr(4 - RegExp.$1.length)); 
    }
     for(var k in o) {
        if(new RegExp("("+ k +")").test(fmt)){
             fmt = fmt.replace(RegExp.$1, (RegExp.$1.length==1) ? (o[k]) : (("00"+ o[k]).substr((""+ o[k]).length)));
         }
     }
    return fmt; 
};
var duration = 17.5;
var sameple_name = "lambda = 1时的混沌体系的增强学习训练";
var note = "在Gray-Scott混沌体系中，当lambda=1时的增强学习训练";
var date  = new Date();
date.setDate(date.getDate() );
var test_date = date.format("yyyy-MM-dd");
var tester_lists = [
    ["BA16003005","刘心爽","化学物理系","xslaq@mail.ustc.edu.cn","13685516029"],
    ["BA15003006","黄德平","化学物理系","hdeping@mail.ustc.edu.cn","18356020036"],
];

var machines = [
    ["933",3.17,16.8],
    ["931",2.71,14.4],
    ["3502",6.78,44.8],
    ["938",0.45,7.2]
];

var index0 = 0;
var index = 1;
var index2 = 0;
var remark = {
    "input[name='instrument_id']":machines[index0][0],
    "input[name='test_date']":test_date,
    "input[name='duration']":duration,
    "textarea[name='note']":note,
    "input[name='sample_name']":sameple_name,
    "input[name='tester_number']":tester_lists[index][0],
    "input[name='tester_name']":tester_lists[index][1],
    "input[name='tester_group']":tester_lists[index][2],
    "input[name='tester_email']":tester_lists[index][3],
    "input[name='tester_phone']":tester_lists[index][4],
    "input[name='operator_number']":tester_lists[index2][0],
    "input[name='operator_name']":tester_lists[index2][1],

};

for(var key in remark){
    document.querySelector(key).value = remark[key];
}


var unit_power_fee = machines[index0][1];
var unit_test_fee  = machines[index0][2];
var power_fee = (duration*unit_power_fee).toFixed(2);
var test_fee  = (duration*unit_test_fee).toFixed(2);

var texts = {
    'span[data-model="power_fee"]':power_fee,
    'span[data-model="test_fee"]':test_fee
}
for(var key in texts){
    document.querySelector(key).textContent = texts[key];
}

// 福建省教育考试院

var links  = $("a");
var link   = "";
var suffix = "";
for(var i = 0;i<links.length;i++){
    link   = links[i].href;
    suffix = link.split(".");
    suffix = suffix[suffix.length - 1];
    if("html" == suffix){
        console.log(i,link,links[i].title);
    }
}

// 中国地情网 http://www.zhongguodiqing.cn
var cities = $(".city");
var city;
var total = {};
for(var i = 0;i < cities.length; i++){
    city = cities[i].childNodes[3];
    total[city.childNodes[0].alt] = city.href;
    console.log(city.childNodes[0].alt,city.href);
    
}

// 搜狗，微信
var boxes = document.querySelectorAll(".txt-box");
var links = [];
var title = "";
for(var i = 0; i < boxes.length; i ++)
{   
    title = boxes[i].querySelector("div>a").textContent;
    if (title == "伯凡时间") {
        links.push(boxes[i].querySelector("h3>a").href);
    }
    
}
var other_pages = document.querySelectorAll("#pagebar_container>a");
var num = other_pages.length - 1;
other_pages[num].click();

// 豆瓣
var text = document.querySelector(".stream-items").textContent;
exportRaw(prompt("请输入文件名","1.json"),text);
document.querySelector(".next>a").click();