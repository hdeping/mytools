/*

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-05-05 06:13:57
    @project      : tree artist
    @version      : 0.4
    @source file  : run.js

============================

*/
var list = {
    "纹理=？":["坏瓜","触感=？","根蒂=？"],
    "触感=？":["坏瓜","好瓜"],
    "根蒂=？":["坏瓜","色泽=？","好瓜"],
    "色泽=？":["好瓜","触感=？","坏瓜"],
    "触感=？1":["坏瓜","好瓜"]
};
// var parents = ["纹理=？","触感=？","根蒂=？",
//                "色泽=？","触感=？"];
var parents = ["触感=？","根蒂=？",
               "色泽=？","触感=？1"];

// var data = {};
var data = get_json("纹理=？");
console.log(data);



// get json data recursively
function get_json(key){
    var data = {};
    data["name"] = key;
    var value = list[key];
    var children = [];
    for(var i = 0, length1 = value.length; i < length1; i++){
        console.log(i,value[i] in parents,parents);
        
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
