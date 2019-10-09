/*

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-05-05 06:54:43
    @project      : tree artist
    @version      : 0.5
    @source file  : run.js

============================

*/
var list = {
    "1":["坏瓜","2","3"],
    "2":["坏瓜","好瓜"],
    "3":["坏瓜","4","好瓜"],
    "4":["好瓜","5","坏瓜"],
    "5":["坏瓜","好瓜"]
};
var link_texts = {
    "1":["模糊","稍糊","清晰"],
    "2":["软黏","硬滑"],
    "3":["硬挺","稍蜷","蜷缩"],
    "4":["浅白","乌黑","青绿"],
    "5":["软黏","硬滑"]
};

// var parents = ["纹理","触感","根蒂",
//                "色泽","触感"];
// var parents = {"2","3",'4',"5"};
var parents = {
    "1":"纹理=？",
    "2":"触感=？",
    "3":"根蒂=？",
    "4":"色泽=？",
    "5":"触感=？"
};

// var data = {};
var data = get_json("1","");
console.log(data);



// get json data recursively
function get_json(key,link_text){
    var data = {};
    data["name"] = parents[key];
    data["link"] = link_text;
    var value = list[key];
    var children = [];
    // for(var i = 0, length1 = value.length; i < length1; i++){
    for(var length1 = value.length,i = length1-1; i >= 0; i--){
        // console.log(i,value[i],value[i] in parents,parents);
        
        if (value[i] in parents) {
            // get a dict recursively here
            children.push(get_json(value[i],
                                   link_texts[key][i]));
        }
        else{
            children.push({name:value[i],
                           link:link_texts[key][i]});
        }
    }
    data["children"]  = children;
    
    return data;
    
}
