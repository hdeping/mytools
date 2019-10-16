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
from mytools import MyCommon

class CrawlRecipes(MyCommon):
    """
    crawl recipes from meishij.net
    method writeJson was heritated from MyCommon
    """
    def __init__(self):
        super(CrawlRecipes, self).__init__()
        self.baseurl = "https://www.meishij.net/china-food/caixi"
        self.cuisine = {"chuancai":56,"xiangcai":14,"yuecai":24,
                        "dongbeicai":11,"lucai":21,"zhecai":11,
                        "sucai":15,"qingzhencai":5,"mincai":17,
                        "hucai":13,"jingcai":12,"hubeicai":7,
                        "huicai":7,"yucai":5,"xibeicai":9,"yunguicai":5,
                        "jiangxicai":3,"shanxicai":6,"guangxicai":1,
                        "gangtaicai":4,"other":56}
    def getRecipes(self,cuisineName, cuisineUrl, pageNum):
        '''
        1) fench all recipes urls for each cuisine
        2) fench all ingredients for each recipe
        3) write all results for each cuisine
        Input: String cuisineName, String cuisineUrl, int pageNum
        OutPut: txt
        '''
        
        data = {}
        for i in range(1, pageNum + 1):
            url = cuisineUrl + "/" + "?&page=" + str(i)
            page = requests.Session().get(url)
            tree = html.fromstring(page.text)
            regex = '//div[contains(@class, "listtyle1_list clearfix")]/div/a/@%s'
            recipeName = tree.xpath(regex % ("title"))
            recipeUrl  = tree.xpath(regex % ("href"))
            length = len(recipeName)
            for ii in range(length):
                name = recipeName[ii]
                url = recipeUrl[ii]
                [ingredient, value] = self.getIngredients(url)
                # fp.write( ",".join([name] + ingredient + value ) )
                # fp.write("\n") #
                res = {}
                res["url"] = url
                res["ingredient"] = ingredient 
                res["value"] = value 
                data[name] = res
                print("%d-%d/%d: %s"%(i,ii,length,name))
                
        resFile = cuisineName + ".json"
        self.writeJson(data,resFile)

        return
    def getIngredients(self,recipeUrl): 
        '''
        fench ingredients and its value for each recipe
        Input: String recipeUrl
        Output: [ingredient, value]
        '''
        page=requests.Session().get(recipeUrl)
        tree=html.fromstring(page.text)
        # '//div[contains(@class, "yl zl clearfix")]/ul/li//h4/a/text()'
        # '//div[contains(@class, "yl zl clearfix")]/ul/li//h4/span/text()'
        # '//div[contains(@class, "yl fuliao clearfix")]/ul/li//h4/a/text()'
        # '//div[contains(@class, "yl fuliao clearfix")]/ul/li/span/text()'
        regex = '//div[contains(@class, "yl %s clearfix")]/ul/li/%s/text()'
        classTypes = ["zl","fuliao"]
        itemTypes = ["/h4/a","/h4/span","span"]
        ingredient = tree.xpath(regex % (classTypes[0], itemTypes[0])) 
        value = tree.xpath(regex % (classTypes[0], itemTypes[1]))
        ingredient += tree.xpath(regex % (classTypes[1], itemTypes[0])) 
        value += tree.xpath(regex % (classTypes[1], itemTypes[2]))
        # print(ingredient,value)
        return [ingredient, value]
    def run(self):
        """
        docstring for run
        run the spider
        """
        
        total = 0
        for (name, page) in self.cuisine.items():
            url = self.baseurl + "/" + name
            total += page
            print(name,page)
            self.getRecipes(name, url, page)
        print(total)

        return
    def analyzeData(self):
        """
        read data from the json files
        :return: None
        """
        self.dirs = "../../data/cuisine/"
        results = []
        for key in self.cuisine:
            filename = key + ".json"
            data = self.loadJson(self.dirs + filename)
            # print(len(data))
            ingredients = self.getIngredients(data)
            results.extend(ingredients)
        print(len(results))
        print(results)


        return

    def getIngredients(self,data):
        """
        get all the ingredients from data
        :param data: dicts type
        :return: array type, all the ingredients
        """
        results = []
        for key in data:
            ingredients = data[key]["ingredient"]
            results.extend(ingredients)

        return results
    
    
spider = CrawlRecipes()
# spider.run()
spider.analyzeData()

