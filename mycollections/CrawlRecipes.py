#!/usr/local/bin/python3
# -*- coding: UTF-8 -*-
 
"""

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-16 15:03:32
    @project      : fetch all recipes from meishij.net
    @version      : 1.0
    @source file  : CrawlRecipes.py

============================
"""

import requests
from lxml import html

class CrawlRecipes():
    """
    crawl recipes from meishij.net
    """
    def __init__(self):
        super(CrawlRecipes, self).__init__()
        self.baseurl = "https://www.meishij.net/china-food/caixi"
        

    def getRecipes(self,cuisineName, cuisineUrl, pageNum):
        '''
        1) fench all recipes urls for each cuisine
        2) fench all ingredients for each recipe
        3) write all results for each cuisine
        Input: String cuisineName, String cuisineUrl, int pageNum
        OutPut: txt
        '''
        resFile = cuisineName + ".txt"
        fp = open(resFile, "w")
        for i in range(1, pageNum + 1):
            url = cuisineUrl + "/" + "?&page=" + str(i)
            page = requests.Session().get(url)
            tree = html.fromstring(page.text)
            regex = '//div[contains(@class, "listtyle1_list clearfix")]/div/a/@%s'
            recipeName = tree.xpath(regex % ("title"))
            recipeUrl  = tree.xpath(regex % ("href"))
            for i in range(len(recipeName)):
                name = recipeName[i]
                url = recipeUrl[i]
                [ingredient, value] = self.getIngredients(url)
                fp.write( ",".join([name] + ingredient + value ) )
                fp.write("\n") #
        fp.close()


    def getIngredients(self,recipeUrl): 
        '''
        fench ingredients and its value for each recipe
        Input: String recipeUrl
        Output: [ingredient, value]
        '''
        page=requests.Session().get(recipeUrl)
        tree=html.fromstring(page.text)
        ingredient = tree.xpath('//div[contains(@class, "yl zl clearfix")]/ul/li//h4/a/text()') 
        value = tree.xpath('//div[contains(@class, "yl zl clearfix")]/ul/li//h4/span/text()')
        print(ingredient,value)
        ingredient += tree.xpath('//div[contains(@class, "yl fuliao clearfix")]/ul/li//h4/a/text()') 
        value += tree.xpath('//div[contains(@class, "yl fuliao clearfix")]/ul/li/span/text()')
        print(ingredient,value)
        return [ingredient, value]

    def run(self):
        """
        docstring for run
        run the spider
        """
        cuisine = {"chuancai":56,"xiangcai":14,"yuecai":24,
                   "dongbeicai":11,"lucai":21,"zhecai":11,
                   "sucai":15,"qingzhencai":5,"mincai":17,
                   "hucai":13,"jingcai":12,"hubeicai":7,
                   "huicai":7,"yucai":5,"xibeicai":9,"yunguicai":5,
                   "jiangxicai":3,"shanxicai":6,"guangxicai":1,
                   "gangtaicai":4,"other":56}

        
        total = 0
        for (name, page) in cuisine.items():
            url = self.baseurl + "/" + name
            total += page
            print(name,page)
            self.getRecipes(name, url, page)
        print(total)
    
    
spider = CrawlRecipes()
spider.run()


