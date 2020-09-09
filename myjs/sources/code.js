/*

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-09-09 16:56:35
    @project      : show the code with prism.js
    @version      : 1.0
    @source file  : code.js

============================

*/


var count =  0;
var style_count = 0;
var code = document.querySelector("code");
var title = document.querySelector("h3");
var h2 = document.querySelector("h2");
var num = sources.keys.length;
var page = document.querySelector("#page");
change_content();
function next () {
      count += 1;
      if (count >= num) {
          count -= num ;
      }
      change_content();

};
function last () {
    count -= 1;
    if (count < 0) {
        count += num ;
    }
    change_content();
};
function goto () {
    count = parseInt(page.value) % num;
    change_content();
};
function toggle_style () {
    style_count += 1;
    if (style_count >= styles.length) {
        style_count -= styles.length;
    }
    document.head.querySelector("link").href = styles[style_count];
};
function change_content () {
    var value = sources.keys[count].split("/");
    var len = value.length;
    value = value[len-2]+"/"+value[len-1];
    h2.textContent = count + "/" + num;
    title.textContent = value;
    page.value = count;
    code.textContent = sources.values[count];
    Prism.highlightAll();
    window.scrollTo(0,0);
};
    