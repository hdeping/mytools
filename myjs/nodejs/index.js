const express = require('express');
const app = express();

console.log("get jobs info of 225");
var jobs = require("./server.js");


var child_process = require("child_process");

app.set("view engine","ejs");
// static files

app.use(express.static('views'));


app.get('/', function (req, res) {
    // res.send('Hello World!');
    // res.render("index.ejs");
    // var file="/home/hdeping/c/14_javascript/08_nodejs/06_v6/views/index.html";
    // var file="/home/hdp/27_nodejs/03_try/01_v1/views/index.html";
    // res.send(output);
    // res.sendFile(file);
    console.log("refreshing");
    jobs.get_jobs();
    console.log("refresh");
    res.render("index.html");
});

app.post('/refresh', function (req, res) {
    console.log("refreshing");
    jobs.get_jobs();
});
    
// });
// app.get('/refresh', function (req, res) {
//     res.send("hello");
//     res.end;
//     console.log("refreshing");
//     jobs.get_jobs();
//     // res.render("index.ejs");

// });



app.listen(3000, function () {
  console.log('Example app listening on port 3000!');
});