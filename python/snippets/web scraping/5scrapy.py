#Create project on the command line with:
#scrapy startproject project_name
#This creates a folder structure, configure the spider with the settings.py file
#For developing purposes, set CONCURRENT_REQUESTS = 1

from scrapy.selector import Selector
from scrapy.http import HtmlResponse

response = HtmlResponse(url='http://my.domain.com', body='<html><body><h1>Hello Selectors!</h1></body></html>', encoding='UTF-8')
print(Selector(response=response).css('h1::text').extract()) #['Hello Selectors!']

#Generate spider. Type on the command line:
#scrapy genspider spider_name 'www.website.com/folder/whatever/'

#Run the shell with:
#scrapy shell
#>>> fetch('http://website.com/folder/whatever')
#>>> urls = response.css('ul.categories.departments > li > a::atrr(href)').extract()

#In the file spider_name.py
def parse(self, response):
    urls = response.css('ul.categories.departments > li > a::atrr(href)').extract()
    for url in urls:
        yield Request(url, callback=self.parse_department_pages)