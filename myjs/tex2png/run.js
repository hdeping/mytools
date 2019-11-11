/*

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-04-26 20:42:30
    @project      : tex2png
    @version      : 0.9
    @source file  : run.js

============================

*/

function update_formula () {
    var formula    = document.getElementById('formula');
    var width      = document.getElementById('input-width');
    var height     = document.getElementById('input-height');
    var fontsize   = document.getElementById('input-font-size');
    var lineheight = document.getElementById('input-line-height');

    width      = width.value;
    height     = height.value;    
    fontsize   = fontsize.value;  
    lineheight = lineheight.value;


    // console.log(width,height,fontsize,lineheight);
    formula.style.width      = width + "px";
    formula.style.height     = height + "px";
    formula.style.fontSize   = fontsize + "px";
    formula.style.lineHeight = lineheight + "px";


}

$("#get").click(function(event) {
    var text = $("#text1").val();
    text = "$$" + text + "$$";
    $("#formula").text(text);
    MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
    // var math_src = "http://210.45.125.225/mathjax/MathJax.js?config=TeX-AMS-MML_HTMLorMML";
    // $.getScript(math_scr);
    // window.location.reload();
    var convertMeToImg = $('#formula');

    html2canvas(convertMeToImg).then(function(canvas) {
        $('#block').append(canvas);
    });
});

// 1 to 9
var colors = ["","black","white","red","green","blue",
              "cyan","purple","pink"];
var color_types = ["background","color"]

for(var i = 1; i < 10; i ++)
{
    var id = "#li" + i;
    var items = [];
    items.push(".ulbackground>"+id);
    items.push(".ulcolor>"+id);
    for(var ii = 0; ii < 2; ii ++)
    {
        $(items[ii]).css("background",colors[i]);
    }
    // click function
    // background color
    $(items[0]).click(function(event) {
        // console.log(color_types[ii],colors[i]);
        // console.log($(this).css("background"));
        var bg = $(this).css("background")
        console.log("background",bg);
        
        $("#formula").css("background",bg);
    });
    // text color
    $(items[1]).click(function(event) {
        // console.log(color_types[ii],colors[i]);
        // console.log($(this).css("background"));
        var bg = $(this).css("background")
        console.log(bg);
        
        bg = bg.split(' none');
        
        
        // $("#formula").css("color",bg);
        $("#formula").css("color",bg[0]);
        // console.log("color",bg[0]);
        // MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
    });
    
    
}
// clear the text area
$("#clear").click(function(event) {
    // $("#text1").text("");
    $("#text1").val("");
});




