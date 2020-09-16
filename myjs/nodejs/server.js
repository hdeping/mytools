String.prototype.myReplace=function(f,e){
    // regular expression
    var reg=new RegExp(f,"g"); 
    return this.replace(reg,e); 
}


function get_jobs()
{

    var child_process = require('child_process'); 
    var command1 = "q2"
    var command2 = "ssh ncl@210.45.125.100 'rem2'"
    var command3 = "ssh ncl@222.195.94.17 '/home/ncl/shell/jobs'"
    var command4 = "ssh ncl@222.195.94.7 '/home/ncl/shell/jobs'"
    var commands = [command1,command2,command3,command4];
    var outputs = [];
    for(var i = 0; i < 4; i++){
        // console.log(commands[i]);
        
        var output = child_process.execSync(commands[i]);
        output = output.toString();
        outputs.push(output);
    
    }
    // output to json

    
    var json_string = get_json(outputs);

    var fs = require('fs');
    
    
    // console.log(outputs,outputs.length);

    // for(var i = 0;i < 14;i++)
    // {
    //     console.log(i,output.substring(2*i,2*i+1));
    //     
    // }
    
    
    
    fs.writeFile("views/jobs.json",json_string, 
    function(err) {
       if (err) {
           return console.error(err);
       }
    });
    
}
function get_json(outputs){

    var json_string = '{"';
    var keys = [];
    keys.push("CPU Server :: 210.45.125.225");
    keys.push("GPU Server :: 210.45.125.100");
    keys.push("CPU Server :: 222.195.94.17 ");
    keys.push("CPU Server :: 222.195.94.7 ");


    json_string += keys[0] + '":';
    json_string += get_225(outputs[0]);
    json_string += ',"'+keys[1]+'":';
    json_string += get_100(outputs[1]);
    json_string += ',"'+keys[2]+'":';
    json_string += get_17(outputs[2]);
    json_string += ',"'+keys[3]+'":';
    json_string += get_7(outputs[3]);
    json_string += "}";

    console.log(json_string);


    return json_string;

}
// for 225 server
function get_225(output){
    var res = '[';
    output = output.myReplace('\n',',');
    output = output.substring(0,output.length-1);
    res += output + ']';
    return res;

}

// for gpu server
function get_100(output){
    var res = {};
    output = output.myReplace('\n',',');
    output = output.substring(0,output.length-1);
    output = output.split(",");
    var length = Math.floor(output.length/3);
    for(var i = 0;i < length; i++){

        var jobs = []
        jobs.push(new Number(output[3*i+1]));
        jobs.push(new Number(output[3*i+2]));
        res[output[3*i]] = jobs;
    
    }
    
    res['remain'] = new Number(output[output.length-1]);
    res = JSON.stringify(res);
    // console.log(res);
    return res;
}
// for .17 server
function get_17(output){
    var res = {};
    output = output.myReplace('\n','/');
    output = output.substring(0,output.length-1);
    output = output.split("/");
    var length = Math.floor(output.length/3);

    var keys = ["node1","node10"];
    for(var i = 2;i < length; i++){
        keys.push("node"+i);
    }
    for(var i = 0;i < length; i++){
        var jobs = []
        jobs.push(new Number(output[3*i+1]));
        jobs.push(new Number(output[3*i+2]));
        jobs[1] -= jobs[0];
        res[keys[i]] = jobs;
    }
    // console.log(keys);


    res = JSON.stringify(res);
    return res;

}
// for .7 server
function get_7(output){
    var res = {};
    output = output.myReplace('\n','/');
    output = output.myReplace(' ','/');
    output = output.substring(0,output.length-1);
    output = output.split("/");
    var length = Math.floor(output.length/4);

    var keys = ["node1","node2","node11","node12",];
    for(var i = 3;i < 11; i++){
        keys.push("node"+i);
    }
    for(var i = 0;i < length; i++){
        var jobs = []
        jobs.push(new Number(output[4*i+1]));
        jobs.push(new Number(output[4*i+2]));
        jobs.push(output[4*i+3])
        jobs[1] -= jobs[0];
        res[keys[i]] = jobs;
    }
    // console.log(keys);

    res["remain"] = new Number(output[output.length-1])


    res = JSON.stringify(res);
    return res;

}
get_jobs();
module.exports = {
    get_jobs
}