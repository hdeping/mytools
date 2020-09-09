function sleep (time) {
  return new Promise((resolve) => setTimeout(resolve, time));
};

var count = 0;
(async function  () {
    var button;
    for(var i = 0; i < 10; i ++)
    {
        button = document.querySelectorAll(".links");
        button[3].childNodes[5].click();
        count += 1;
        console.log(count);
        await sleep(1000);
    }
})();
