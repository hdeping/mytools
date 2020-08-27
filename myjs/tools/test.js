/*

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-08-07 17:35:01
    @project      : test the input
    @version      : 1.0
    @source file  : test.js

============================

*/

var vm = new Vue({
        el:"#app",
        data:{
            items:[
            {
                field:"name",
                name:"name"
            },
            {
                field:"gender",
                name:"gender"
            },
            {
                field:"age",
                name:"age"
            },
            {
                field:"favorite",
                name:"favorite"
            }
            ],
            variables:[
                "Q",
                "$S_0$",
                "$X_{0,i}$",
                "II",
                "XT",
                "SRT",
                "k",
                "Y",
                "$k_d$",
                "$K_s$",
                "$f_d$"
            ],
            refs:[
                "i0","i1","i2","i3","i4",
                "i5","i6","i7","i8","i9",
                "i10"
            ],
            values:[
                1000,192,30,10,
                2500,6,12.5,0.40,
                0.10,10,0.15
            ]
        },
        methods:{
            calculate:function  (event) {

                var Q   = parseFloat(this.$refs.input[0].value);
                var S0  = parseFloat(this.$refs.input[1].value);
                var X0i = parseFloat(this.$refs.input[2].value);
                var II  = parseFloat(this.$refs.input[3].value);
                var XT  = parseFloat(this.$refs.input[4].value);
                var SRT = parseFloat(this.$refs.input[5].value);
                var k   = parseFloat(this.$refs.input[6].value);
                var Y   = parseFloat(this.$refs.input[7].value);
                var kd  = parseFloat(this.$refs.input[8].value);
                var Ks  = parseFloat(this.$refs.input[9].value);
                var fd  = parseFloat(this.$refs.input[10].value);
                // alert("Are you ok?")
                var S = Ks*(1+kd*SRT)/(SRT*(Y*k-kd)-1);
                var HRT = SRT/XT*(Y*(S0-S)*(1+fd*kd*SRT)/(1+kd*SRT)+X0i);
                var PxTVSS = (XT*HRT*Q/SRT)/1000;
                var PxTTSS = ((Q*Y*(S0-S)/(1+kd*SRT)/0.85+fd*kd*Y*Q*(S0-S)*SRT/(1+kd*SRT)/0.85+Q*X0i+Q*(II)))/1000;

                this.$refs.output1.value = PxTVSS.toFixed(2);
                this.$refs.output2.value = PxTTSS.toFixed(2);
                
            },
            reset:function  (event) {
                for(var i = 0; i < 11; i ++)
                {
                    this.$refs.input[i].value = parseFloat(this.$refs.input[i].value)+1;
                }
                
            }

        }
    });

