javascript:
function sleep (time) {
  return new Promise((resolve) => setTimeout(resolve, time));
}
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

var data = {
    "zxi":{
        "类型":"团体",
        "名称":"贵州鑫鼎钰典当有限公司",
        "证件":"91520102MA6GRU2L68",
        "联系人":"张鑫"
    },
    "zly":{
        "类型":"个人",
        "名称":"张磊毅",
        "证件":"33070319910426003X"
    },
};

function fillForm (strings) {
(async function() {
    var email = "hyxbdcs@163.com";
    document.querySelector("#email").value = email;
    var insure_types = {};
    insure_types["个人"] = [1,"01","none"];
    insure_types["团体"] = [2,"09","block"];
    
    var length = 5;
    var checkBox = document.querySelector("#page_insured_related");
    if (strings.length > 4 && strings.length != length) {
        alert("字段数量超过4个，但不是5，请仔细检查");
    }
    else if (strings.length == length) {
        document.querySelector("#sameAsAppliCheckBox").click();
        await sleep(1000);
        if (!(strings[4] in data)) {
            alert(strings[4] + "不在投保人列表中，请在data中完善相关信息！");
        }
        var item = data[strings[4]];
        var remark = ['select[name="nbzRiskRelatedPartyVo.insuredType"]',
                      'input[name="nbzRiskRelatedPartyVo.insuredName"]',
                      'select[name="nbzRiskRelatedPartyVo.identifyType"]',
                      'input[name="nbzRiskRelatedPartyVo.identifyNumber"]',
                      'input[name="nbzRiskRelatedPartyVo.contactName"]'];
        var specie = item["类型"];
        document.querySelector(remark[0]).value = insure_types[specie][0];
        document.querySelector(remark[2]).value = insure_types[specie][1];
        var contact = document.querySelectorAll("#div_contact")[1];
        if (specie == "个人") {
            contact.style.display = insure_types[specie][2];
        }
        else if (specie == "团体") {
            contact.style.display = insure_types[specie][2];
            document.querySelector(remark[4]).value = item["联系人"];
        }
        else{
            alert("被保险人类型只能是个人或者团体，请在data中检查相关信息！");
        }
        document.querySelector(remark[1]).value = item["名称"];
        document.querySelector(remark[3]).value = item["证件"];

    }
    var date = new Date();
    date.setMinutes(date.getMinutes() + 5);
    var begin = 0;
    var from = strings[begin]; 
    var to = strings[begin+1]; 
    var package = strings[begin+2];
    var check = strings[begin+3];
    var serial = package;
    var insurance = check;
    var remark = {
        "#GuItemDynamicFieldAH":date.format("yyyy-MM-dd hh:mm:ss"),
        "#GuItemDynamicFieldAL":from,
        "#GuItemDynamicFieldAN":to,
        "#GuItemDynamicFieldAQ":"由" + from + "至" + to,
        "#GuItemDynamicFieldBC":package,
        "#GuItemDynamicFieldBF":check,
        "#GuItemDynamicFieldBH":serial,
    };
    for(var key in remark){
        document.querySelector(key).value = remark[key];
    }
                   
    document.querySelector("#bt_tablerow_insert").click();

})();
};

var strings = prompt("请输入字段信息","启始地 目的地 运单号 保额 (被保险人) (用空格隔开)");
strings = strings.split(" ");

if (strings.length < 4) {
    alert("字段数量不足4个，或字段未用英文逗号隔开");
}
else {
    fillForm(strings);
}