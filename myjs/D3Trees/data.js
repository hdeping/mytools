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
var tran = {
    "1":"纹理=？",
    "2":"触感=？",
    "3":"根蒂=？",
    "4":"色泽=？",
    "5":"触感=？"
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
var data = get_json("1");
console.log(data);



// get json data recursively
function get_json(key){
    var data = {};
    data["name"] = tran[key];
    var value = list[key];
    var children = [];
    // for(var i = 0, length1 = value.length; i < length1; i++){
    for(var length1 = value.length, i = length1 - 1; i >= 0; i--){
        // console.log(i,value[i],value[i] in parents,parents);
        
        if (value[i] in parents) {
            // get a dict recursively here
            children.push(get_json(value[i]));
        }
        else{
            children.push({name:value[i]});
        }
    }
    data["children"]  = children;
    
    return data;
    
}
