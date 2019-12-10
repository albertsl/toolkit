from urllib.request import urlopen
import re

def download_page(url):
    return urlopen(url).read()#.decode('utf-8')

def download_and_save_page(url, file):
    html_code = urlopen(url).read()#.decode('utf-8')
    f = open(file, 'wb')
    f.write(html_code)
    f.close()
    return html_code

def extract_links(page):
    link_regex = re.compile('<a[^>]+href=["\'](.*?)["\']', re.IGNORECASE)
    return link_regex.findall(page)

if __name__ == '__main__':
    target_url = 'http://www.apress.com'
    links = extract_links(download_page(target_url))