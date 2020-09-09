#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-06-06 10:50:07
    @project      : use selenium to get data
    @version      : 1.0
    @source file  : mahua.py
                    Spider.py

============================
"""

from selenium import webdriver
from time import sleep
import scrapy
from mytools import MyCommon
import os
import numpy as np
import platform
from tqdm import tqdm
import numpy as np
from selenium.webdriver.firefox.options import Options 
from selenium.webdriver.common.action_chains import ActionChains
import numpy as np


class Spider(MyCommon):
    """docstring for Spider"""
    def __init__(self):
        super(Spider, self).__init__()
        self.formats = []
        self.formats.append("<a href='%s'><h1>%s</h1></a>\n")
        self.formats.append('<img src="%s">\n')
        self.preamble = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <title> Images </title>
        </head>
        <body>
        %s 
        </body>
        </html>
        """

        self.backRun = "nohup %s 1> /dev/null 2> /dev/null &"

        self.video_format = """
<video             
    src="%s"    
    volume="20"      
    controls       
    width="720"      
    height="480"     
    type="video/mp4">
</video>
        """

        self.h1 = """
<h1> %s </h1>
        """

        self.href = """
<a href="%s"> %s </a>
        """

        self.m3u8 = """
#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:8
#EXT-X-MEDIA-SEQUENCE:0
%s
#EXT-X-ENDLIST
        """
        return 

    def initDriver(self):
        """
        docstring for initDriver
        """
        print("initialize the webdriver")
        sys = platform.system()
        options = Options()
        options.add_argument("--headless")
        if sys == "Linux":
            self.bro = webdriver.Firefox(options=options)
        else:
            self.bro = webdriver.Safari()
        return

    def getImages(self):
        urls = self.loadStrings("urls.json")
        # print(urls)
        bro = webdriver.Safari()
        strings = ""

        

        for i,url in enumerate(urls):
            print("getting ",i)
            sleep(1)
            bro.get(urls[i])
            html = bro.page_source
            data = scrapy.Selector(text=html)

            fields = ["//h2/text()","//img[@class='lazy']/@src","//li/input/@value"]
            texts = []
            for field in fields:
                title = data.xpath(field).extract()[0]
                texts.append(title)
            tmp = ""
            tmp += self.formats[0]%(texts[2],texts[0])
            tmp += self.formats[1]%(texts[1])
            strings += tmp
            print(tmp)


        filename = "total.html" 

        with open(filename,"w") as fp:
            print("write to",filename)
            fp.write(self.preamble%strings)
        bro.quit()
        return
        
    def getUrlPages(self):
        bro = webdriver.Safari()
        prefix = "http://www.mahuazy.com/"
        url = prefix + "?m=vod-type-id-19-pg-%d.html"
        pages = 17
        for i in range(3,pages+1):
            print("getting ",i)
            bro.get(url%(i))
            sleep(2)

            html = bro.page_source
            data = scrapy.Selector(text=html)
            field = "//span[@class='xing_vb4']/a/@href"
            urls = data.xpath(field).extract()
            for link in urls:
                print(prefix+link)

            print("finishing")
            sleep(1)

        bro.quit()

    def getJPages(self):

        pre2 = "https://jj.002xf.com/vodshow/6--------%d---.html"
        # get pages
        pages = []
        fields = ["//ul/li/em/a/@href","//ul/li/em/a/img/@src"]
        images = []

        filename = "pages.json"
        if not os.path.exists(filename):
            for i in range(9,80):
                src,img = self.getContent(pre2%i,fields)
                print(i,len(src),len(img))
                for ch in src:
                    ch = ch.split("/")[-1]
                    ch = ch.split(".")[0]
                    pages.append(int(ch))
                if i == 3:
                    break
            print(len(pages))
            items = []
            for i in pages:
                if i not in items:
                    items.append(i)
            pages = items
            print(len(pages))
            self.writeJson(pages,filename)
        else:
            pages = self.loadJson(filename)
        return pages

    def getJJRF(self):
        self.initDriver()
        pages = [7333,7496,10462]
        prefix = "https://jj.002xf.com/play/%d/3/1.html"

        pages = self.getJPages()

        fields = ["//ul/li/text()","//td/iframe/@src"]
        lines = ""
        count = 0
        for i in pages:
            count += 1
            url = prefix % (i)
            
            title,url = self.getContent(url, fields)
            print("count : ",count)
            if len(url) > 0:
                url = url[0].split("=")[1]
                strings = self.formats[0]%(url,title[0])
                print(strings)
                lines += strings+"\n"
            else:
                print("no m3u8")

        with open("l2.html","w") as fp:
            fp.write(self.preamble%(lines))



        self.exitBrowser()


    def get845665(self):
        """

        """
        self.initDriver()
        url = "https://www.845465.com/#/vodAllData?id=12&pid=1&title=日本无码"
        urls = [url]
        fields = ["//div[@aria-label='Video Player']/@src"]
        fields = ["//ul/li/img/@src"]
        for url in urls:
            print(url)
            line = self.getContent(url,fields)
            print(line)

        # https://www.845447.com/xmkk/ed0f3a30/m3u8.m3u8
        # https://www.845447.com/xmkk/acf15aad/m3u8.m3u8

        self.exitBrowser()
        return 

    def exitBrowser(self):
        """
        exit the webdriver
        """
        print("exit the browser")
        self.bro.quit()
        return 

    def getContent(self,url,fields):
        """
        get the content under the field of a specific url
        """
        self.bro.get(url)
        sleep(np.random.rand()*4.5 + 0.5)
        content = self.analyzePage(fields)
        return content

    def analyzePage(self,fields):
        """
        docstring for analyzePage
        """
        html = self.bro.page_source
        data = scrapy.Selector(text=html)
        content = []
        for field in fields:
            content.append(data.xpath(field).extract())
        return content

    def getURLs(self,urlType=1,arr=None):
        """
        docstring for getURLs
        """
        urls   = []
        if urlType == 1:
            prefix = "http://www.ttbcj.cn/video/%d.html?%d-0-0"
            if arr == None:
                arr = [23580,23576,23582,23325,23572]
        else:
            urls   = ["http://www.ttbcj.cn/list/index36.html"]
            prefix = "http://www.ttbcj.cn/list/index36_%d.html"
            arr = np.arange(2,82)

        for i in arr:
            if urlType == 1:
                urls.append(prefix % (i,i))
            else:
                urls.append(prefix % (i))
        return urls

    def getJs(self):
        """
        docstring for getJs
        """
        self.initDriver()
        pages = self.loadJson("pages1.json")
        urls = self.getURLs(arr=pages)
        fields = ["//script/@src"]
        fields = ["//div[@class='player']/div/script/@src"]
        prefix = "http://www.ttbcj.cn"
        for url in tqdm(urls):
            contents = self.getContent(url, fields)
            if len(contents[0]) > 0:
                line = contents[0][0]
                line = line.split("?")[0]
                command = "wget %s%s"%(prefix,line)
                print(command)
                os.system(command)


        self.exitBrowser()
        return

    def getJs2(self):
        pages = np.loadtxt("pages1.json")
        pages = pages.astype(int)

        prefix = "http://www.ttbcj.cn/video/%d.html?%d-0-0"
        suffix = "1> /dev/null 2> /dev/null &"
        for i in pages:
            break
            url = prefix%(i,i)
            command = "nohup wget %s -O %d.html %s"%(url,i,suffix)
            print(command)
            os.system(command)

        pages = self.loadStrings("playdata.json")
        prefix = "http://www.ttbcj.cn"
        for page in pages:
            url = prefix + page
            command = "nohup wget %s %s"%(url,suffix)
            print(command)
            os.system(command)




    def getJsPages(self):
        """
        docstring for getJsPages
        """
        self.initDriver()
        urls = self.getURLs(urlType=2)
        fields = ["//ul/li/a/@href"]
        prefix = "http://www.ttbcj.cn"
        pages = []
        for url in urls:
            contents = self.getContent(url, fields)
            # print(contents)
            for line in contents[0]:
                line = line.split("/")
                if line[1] == "view":
                    line = line[-1].split("x")
                    line = line[-1].split(".")[0]

                    pages.append(line)
            break

        self.writeJson(pages,"pages1.json")


        self.exitBrowser()
        return
    def dealData(self):
        """
        docstring for dealData
        """
        data = self.loadStrings("mv.js")[0]
        data = data.split("$")
        # print(data)
        for line in data:
            arr = line.split(".")
            if arr[-1] == "m3u8":
                print(line)
        return
    def haimaoba(self):
        """
        docstring for haimaoba
        """
        begin = 68064
        end   = 68073
        prefix = "http://www.haimaoba.com/catalog/4025/%d.html"
        urls = []
        for i in range(begin, end + 1):
            url = prefix%(i)
            print(url)
            os.system("nohup wget %s 1> /dev/null 2> /dev/null &"%(url))
        return
    def downloadHtml(self,prefix,begin,end):
        """
        docstring for downloadHtml
        """
        for i in range(begin, end):
            url = prefix%(i)
            print(url)
            os.system("nohup wget %s 1> /dev/null 2> /dev/null &"%(url))
        return
    def toho(self):
        """
        docstring for toho
        """
        prefix = "http://www.toho5.com"
        self.initDriver()
        fields = ["//div/img/@src"]
        fields.append("//div[@class='mh_headpager tc']/a/@href")
        count = 0

        num = [[22174,0],
               [22399,0,5],
               [21370,0,50],
               [21373,22,47],
               [21245,0,25],
               [15185,0,21],
               [7648,0,38],
               [22868,0,30],
               [14846,0,30],
               [15637,0,10],
               [8201,0,50],
               [15262,0,40],
               [13564,0,56],
               [23408,0,24],
               [23410,11,21],
               []]
        num = num[-2]
        url = "http://www.toho5.com/cartoon14/%d-%d.html"%(num[0],num[1])

        chapter_count = 0
        chapter = num[2]
        strings = ""
        filename = "%d.html"%(num[0])
        while 1:
            contents = self.getContent(url, fields)
            img = "http:" + contents[0][0]
            img = self.formats[1]%img
            strings += img+"\n"
            print(img)
            url = contents[1][2]
            if url[:4] == "java":
                url = contents[1][3]
                chapter_count += 1 
                if chapter_count == chapter:
                    print("finished %d chapters"%(chapter))
                    break
            url = prefix + url
            count += 1
            print(count,url)
            # break
        with open(filename,"w") as fp:
            fp.write(strings)

        self.exitBrowser()

        return

    def toho2(self):
        """
        docstring for toho2
        """
        prefix = "http://www.toho5.com/cartoon14/%d-0.html"
        self.downloadHtml(prefix,5698,5718)
        return
    def getM3u8(self):
        """
        docstring for getM3u8
        """
        url = "https://www.ai66.cc/e/DownSys/play/?classid=20&id=13287&pathid1=0&bf=0"
        self.initDriver()
        fields = ["//iframe/@src"]
        contents = self.getContent(url,fields)
        print(contents)
        self.exitBrowser()
        return  

    def fangzhouzi(self):
        """
        docstring for fangzhouzi
        """
        self.initDriver()
        prefix = "https://www.ximalaya.com"
        url = "/keji/19422814/p%d"
        fields = ["//li/div/a/@title","//li/div/a/@href"]
        fields1 = ["//article[@class='intro  tj_']/text()"]
        print("# 方舟子文章合集")
        count = 0 
        for i in range(8,11):
            count += 1 
            print("count: ",count)
            link = prefix+url%i 
            titles,hrefs = self.getContent(link,fields)
            for title,href in zip(titles,hrefs):
                print("## ",title)
                href = prefix + href
                texts = self.getContent(href,fields1)
                print("\n".join(texts[0]))

        self.exitBrowser()
        return

    def annualReview(self):
        """
        docstring for annualReview
        """
        prefix = "https://www.annualreviews.org/toc/physchem/%d/1"
        for i in range(1,72):
            url = prefix%i 
            command = self.backRun%("wget %s -O %d.html"%(url,i))
            os.system(command)

        return


    def hmb6(self):
        self.initDriver()
        url = "http://hmb6.com/chapter/%d"
        url = "https://www.mm820.com/chapter/%d"
        fields = ["//img/@src"]
        # for i in range(2295,2305):
        links = ""

        prefix = "https://www.mm820.com"
        fields = ["//li/a/@href"]
        begin = 282
        urls = self.getContent(prefix + "/chapter/%d"%begin,fields)[0]
        print(urls)
        fields = ["//div[@class='comicpage']/img/@src","//select/option"]
        for i in urls:
            url = prefix + i
            contents  = self.getContent(url,fields)
            for link in contents[0][:-1]:
                link = self.formats[1]%link
                links += '"%s"\n'%link
                print(link)
            pages = len(contents[1])
            print(i,pages)
            for j in range(2,pages+1):
                print(i,j)
                contents  = self.getContent(url+"?page=%d"%(j),fields)
                for link in contents[0][:-1]:
                    link = self.formats[1]%link
                    links += '"%s"\n'%link 
                    print(link)

        with open("manhua-%d.js"%begin,"w") as fp:
            fp.write("var images = [\n")
            fp.write(links)
            fp.write("]")
        self.exitBrowser()

    def ReinforceLearning2(self):
        """
        docstring for ReinforceLearning
        """
        url = "http://rail.eecs.berkeley.edu/deeprlcourse/"
        self.initDriver()
        fields = ["//li/a/@href"]
        prefix = "http://rail.eecs.berkeley.edu"
        links = self.getContent(url, fields)
        for link in links[0]:
            print(prefix + link)
        self.exitBrowser()
        return

    def ReinforceLearning(self):
        """
        docstring for ReinforceLearning
        """
        prefix = "http://rail.eecs.berkeley.edu"
        bro = self.loadHtml("test.html")
        links = bro.xpath("//li/a/@href").extract()
        for link in links:
            if link.endswith("pdf"):
                print(prefix + link)

        return

    def getSerialName(self,index):
        """
        docstring for getSerialName
        """
        if index < 10:
            name = "000%d"%(index)
        elif index < 100:
            name = "00%d"%(index)
        elif index < 1000:
            name = "0%d"%(index)
        else:
            name = "%d"%(index)
        return name

    def downloadTs(self):
        """
        docstring for downloadTs
        """
        urls = self.loadStrings("cdn-aliyun.m3u")

        video_type = "index%d.ts"
        arr = [100]
        dicts = {}

        contents = ""
        for i,url in enumerate(urls):
            name = self.getSerialName(i+1)
            print(i,name)
            item = {}
            item["url"] = url
            names = []
            contents += self.h1%name
            for j in arr:
                link = url.replace("index.m3u8",video_type%j)
                video_name = "%s-%s"%(name,video_type%j)
                command = "wget %s -O %s"%(link,video_name)
                command = self.backRun%command
                os.system(command)
                # print(command)
                contents += self.video_format%video_name
                names.append(video_name)
            item["name"] = names 
            dicts[name] = item
            
            # if i == 10:
            #     break

        self.writeJson(dicts,"cdn-aliyun.json")

        with open("cdn-aliyun.html","w") as fp:
            fp.write(contents)

            
        return

    def testComMl(self):
        """
        docstring for testComMl
        """
        urls = self.loadStrings("com-ml.m3u")
        urls = self.loadStrings("cdn-aliyun.m3u")
        contents = ""

        for i,url in enumerate(urls):
            name = self.getSerialName(i+1)
            link = self.href%(url,self.h1%name)
            contents += link

            # link = url.replace("index.m3u8","800kb/hls/index.m3u8")
            # command = "wget %s -O %s.m3u8" % (link,name)
            # command = self.backRun%command
            # os.system(command)
            # print(i,command)

        filename = "com-ml.html"
        filename = "ali.html"
        with open(filename,"w") as fp:
            fp.write(contents)
        return 

        prefix = ["https://cdn.com-ml-zy.com",
                  "https://cdn.com-ml-zyw.com"]
        dicts = {}
        for i in range(1,417):
            name = "%s.m3u8"% self.getSerialName(i)
            print(i,name)
            texts = self.loadStrings(name)
            tag = texts[5].split("/")[2]
            dicts[name] = tag
            if len(texts) < 202:
                ii = -7
            else:
                ii = 202
            jj = 1
            if i <= 251:
                jj = 0 
            contents += "%s\n%s%s\n"%(texts[ii],prefix[jj],texts[ii+1])

        self.writeJson(dicts,"l5.json")

        
        with open("l5.m3u8","w") as fp:
            fp.write(self.m3u8%contents)   

        return
    def addItems(self,contents):
        """
        docstring for addItems
        """
        for url,name in zip(contents[0],contents[1]):
            item = {}
            item["name"] = name 
            item["url"]  = url
            self.items.append(item)

        return

    def jianChaYuan(self):
        """
        docstring for jianChaYuan
        """
        url = "https://www.12309.gov.cn/12309/gj/fj/nds/ndsgtx/zjxflws/index.shtml?channelLevels=/fb5a41c9247547bca03ae21326c3ad51/e2d8081e3a3640719cf2b3dedfb39725/13a9e2732a2b4d0b92ecf96a7d0762e4/14ea6c579e264b5495f073b6be4ad19b/32ed02535b6e4cd0a42b8187095f5847"

        self.initDriver()
        line = "//cmspro_documents/li/a"
        fields = ["%s/@href"%line,"%s/text()"%line]
        self.items = []
        contents = self.getContent(url,fields)
        # print(contents)
        self.addItems(contents)
        print(self.items)

        pages = 3
        # nextbtn = "document.getElementByTagName('a').click()"
        nextbtn = "(//div[@id='page_div']/a)[-1]"
        for i in range(pages - 1):
            # go to next page
            print("page ",i+2)
            self.bro.find_element_by_class_name("zxfokbtn").click()
            sleep(2)
            contents = self.analyzePage(fields)
            self.addItems(contents)

        # self.exitBrowser()

        self.writeJson(self.items,"gutian_qisushu.json")
        return

    def testClick(self):
        """
        docstring for testClick
        """
        self.initDriver()
        url = "http://www.7160.com/zhenrenxiu/66829/"

        js = "var q = document.getElementsByClassName('itempage').children[-1].click()"
        js = 'document.getElementsByClassName("itempage")[-1].click();'
        # self.bro.maximize_window()
        self.bro.get(url)
        for i in range(4,50):
            print("page",i)
            button = "(//div[@class='itempage']/a)[%d]"%(-1)
            # button = "(//a/img)[2]"
            # self.bro.find_element_by_xpath(button).click().perform()
             
            button = self.bro.find_element_by_link_text("下一页")
            # button = self.bro.find_element_by_xpath(button)
            self.bro.execute_script("arguments[0].scrollIntoView();", button) 
            # self.bro.execute_script("window.scrollTo(0, 1400)");
            btn = self.bro.find_element_by_link_text("下一页")
            ActionChains(self.bro).move_to_element(button).click(button).perform()
            sleep(2)
        self.exitBrowser()
        return

    def guTianQiSu(self):
        """
        docstring for guTianQiSu
        古田，起诉书
        """
        name = "gutian"
        data = self.loadHtml("%s.html"%name)

        line = "//cmspro_documents/li/a"
        fields = ["%s/@href"%line,"%s/text()"%line]
        self.items = []


        contents = []
        for field in fields:
            contents.append(data.xpath(field).extract())
        # print(contents)

        texts = []
        for line in contents[1]:
            # line = line.encode('utf-8').decode('gbk')
            line = line.encode("ISO-8859-2").decode('utf-8')
            texts.append(line)
        contents[1] = texts

        # self.initDriver()

        total = ""
        fields = ["//div[@class='detail_con']/p/span/text()"]
        for i,url in enumerate(contents[0]):
            break
            print(i,url)
            texts = self.getContent(url,fields)[0]
            total += "\n".join(texts) + "\n"
            print(total)

        with open("%s.txt"%name,"w") as fp:
            fp.write(total)

        # self.exitBrowser()

        self.addItems(contents)
        self.writeJson(self.items,"%s_qisushu.json"%name)



        return

    def getBiQuGe(self):
        """
        docstring for getBiQuGe
        get books from biquge
        """
        prefix = "http://www.biquge.info/"
        url = "%s8_8704/"%prefix
        url = "%s63_63936/"%prefix
        prefix = "http://www.tianxiabachang.cn/"
        url = "%s0_381/"%prefix


        self.initDriver()
        fields = ["//dd/a/text()","//dd/a/@href"]
        contents = self.getContent(url,fields)
        for i,title in enumerate(contents[0]):
            print(i,title,contents[1][i])

        fields = ["//div/h1/text()","//div[@id='content']/text()"]
        # get contents
        num = 0
        total = ""
        for i,url in tqdm(enumerate(contents[1][num:])):
            try:
                url = prefix + url 
                content = self.getContent(url,fields)
                s = content[0][0] + "\n"
                s += "\n".join(content[1]) + "\n"
                total += s
            except Exception:
                print("wrong with ",i,url)


        with open("linyuanxing.txt","w") as fp:
            fp.write(total)
        self.exitBrowser()
        return

    def getM3uLinks(self):
        """
        docstring for getM3uLinks
        """
        indeces = np.loadtxt("l2.sh").astype(int)
        links = self.loadStrings("cdn-aliyun2.m3u")
        print(indeces)
        for i in indeces:
            text = self.href%(links[i-1],self.h1%str(i))
            print(text)
        return

    def zhiyuanInfo(self):
        """
        docstring for zhiyuanInfo
        """
        serials = self.loadJson("serial.json")
        prefix = "https://gkcx.eol.cn/school/%s"
        self.initDriver()
        fields = ["//div[@class='line3_item']/span/a/@href",
                   "//li[@class='item_bottom_li']/span/text()" ]

        data = {}
        for i in serials:
            url = prefix%i
            contents = self.getContent(url,fields)
            intro = "学校网址："+"\n".join(contents[0] + contents[1]) 
            print(intro)
            data[i] = intro 
        
        self.writeJson(data,"xuexiao_intro.json")

        self.exitBrowser()

        return

    def luchanghai(self):
        """
        docstring for luchanghai
        """
        prefix = "https://www.changhai.org"
        url = "https://www.changhai.org/indices/updates.php"
        self.initDriver()
        fields = ["//li/a/@href"]
        contents = self.getContent(url,fields)[0]

        urls = []
        for line in contents:
            link = url + line[2:]
            urls.append(link)

        self.writeJson(urls,"luchanghai_links.json")

        self.exitBrowser()
        return

    def luchanghaiContents(self):
        """
        docstring for luchanghaiContents
        """
        urls = self.loadJson("luchanghai_links2.json")
        self.initDriver()
        fields = ["//td/p/text()"]

        num = len(urls)
        with open("卢昌海.txt","w") as fp:
            for i,url in enumerate(urls):
                fp.write(url+"\n")
                print("serial: %d/%d"%(i+1,num),url)
                contents = self.getContent(url,fields)[0]
                for line in contents:
                    line = line.replace("\n\n","")
                    fp.write(line)
                    print(line)
        self.exitBrowser()



        return


    def pingGuoDianYing(self):
        """
        docstring for pingGuoDianYing
        """
        prefix = "http://www.gooddyw.cc"
        url = "http://www.gooddyw.cc/vodtype/lunli-%d"
        fields = ["//li/div/a/@title","//li/div/a/@href"]
        self.initDriver()

        end = 2 
        for i in range(1,end):
            contents = self.getContent(url%i,fields)
            # print(contents)
            for u,v in zip(contents[0],contents[1]):
                link = "%s/vodplay/%s-1-1"%(prefix,v.split("/")[-2])
                m3u8 = self.getContent(link,["//iframe/@src"])[0]
                if len(m3u8) > 0:
                    print(u,link,m3u8)
                
        self.exitBrowser()
        return

    def getPingGuo(self):
        """
        docstring for getPingGuo
        """
        self.initDriver()
        link = "http://www.gooddyw.cc/vodplay/yemannvwang-1-2/"
        m3u8 = self.getContent(link,["//iframe/@src"])[0]
        print(m3u8)
        self.exitBrowser()
        return

    def getNovel(self):
        """
        docstring for getNovel
        """
        url = "https://www.kele25.com/novel/novel_list.html?novel_type=2&page_index=%d"
        prefix = "https://www.kele25.com"

        arr = []
        self.initDriver()
        for i in range(1,11):
            contents = self.getContent(url%i,["//li/a/@href"])[0]
            arr += contents 
        self.exitBrowser()
        arr = np.flip(arr)

        res = []
        for line in arr:
            link = prefix+line
            print(link)
            res.append(link)

        self.writeJson("novel_links.json",res)

        return

    def getNovelContents(self):
        """
        docstring for getNovelContents
        """
        
        urls = self.loadJson("novel_links.json")
        self.initDriver()
        fields = ["//div/text()"]
        with open("novels.txt","w") as fp:
            for i,url in enumerate(urls):
                print(url)
                contents = self.getContent(url,fields)[0]
                print("".join(contents))
                fp.write("%s\n%s"%(url,"".join(contents)))
        self.exitBrowser()

        return
    def testBefore(self):
        """
        docstring for testBefore
        """
         # self.getImages()
        # self.getJJRF()
        # self.get845665()
        # self.getJs()
        # self.dealData()
        # self.getJsPages()
        # self.getJs2()
        # self.haimaoba()
        # self.toho()
        # self.getM3u8()
        # self.fangzhouzi()
        # self.hmb6()
        # self.annualReview()
        # self.ReinforceLearning()
        # self.downloadTs()
        # self.testComMl()
        # self.testClick()
        # self.jianChaYuan()
        # self.guTianQiSu()
        # self.getBiQuGe()
        # self.getM3uLinks()
        # self.zhiyuanInfo()
        # self.luchanghai()
        # self.luchanghaiContents()
        # self.pingGuoDianYing()
        # self.getPingGuo()
        # self.getNovel()
        # self.getNovelContents()
        return
    def get232y(self):
        """
        docstring for get232y
        """
        url = "https://www.232ys.com/v/%d-1-1.html"
        self.initDriver()
        fields = ["//div[@class='stui-player col-pd']/div/iframe/@src",
                  "//div[@class='title']/a/text()"]

        arr = [56739,60448]
        arr = [40377,2845,2819,4904,2623]
        arr = [6316]
        fp = open("output.json","w")
        for num in arr:
            link = url%num
            contents = self.getContent(link,fields)
            if len(contents[0]) == 1:
                link = contents[0][0].replace("%3A",":")
                link = link.replace("%2F","/")
                if "&" in link:
                    link = link.split("&")[1][4:]
                print(contents[1][0],link)
                fp.write(link+"\n")
        fp.close()


        self.exitBrowser()
        return

    def dealSludge(self):
        """
        docstring for dealSludge
        """
        data = self.loadHtml("sludge.html")
        codes = data.xpath("//script/@src").extract()
        for code in codes:
            print(code)
        return
    def saveContents(self,data,filename):
        """
        docstring for saveContents
        data: array
        filename: string
        """
        with open(filename,'w') as fp:
            for line in data:
                fp.write(line+"\n")

        return
    def po18ks(self):
        """
        docstring for po18ks
        """
        self.initDriver()
        fields = ["//li/a/@href"]
        url = "https://www.po18ks.com/book/19817.html"
        name = url.split("/")[-1][:-5]+".json"
        contents = self.getContent(url, fields)[0]

        fields = ["//div[class='read-content']"]
        novels = []
        for url in contents:
            if "book" in url:
                novel = self.getContent(url, fields)[0]
                # novels.append(novel)
                # print(novel)
                break

        # self.saveContents(novels, name)


        content
        self.exitBrowser()
        return

    def huoxingyizhong(self):
        """
        docstring for huoxingyizhong
        """
        self.initDriver()
        url = "https://manhua.fzdm.com/47/"
        fields = ["//li/a/@href"]
        urls = self.getContent(url,fields)[0]
        urls = np.flip(urls)[:-4]

        fields = ["//img[@id='mhpic']/@src"]
        images = []
        for j,link in enumerate(urls):
            for i in range(60):
                new_link = "%s%sindex_%d.html"%(url,link,i)
                try:
                    images += self.getContent(new_link,fields)[0]
                except Exception:
                    break 
        self.writeJson(images,"images.json")



        self.exitBrowser()
        return
    def test(self):
       
        # self.get232y()
        # self.dealSludge()
        # self.po18ks()
        # self.getBiQuGe()
        self.huoxingyizhong()


        return
