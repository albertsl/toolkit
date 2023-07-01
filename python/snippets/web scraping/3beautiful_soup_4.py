from bs4 import BeautifulSoup

soup = BeautifulSoup(html_code, 'html.parser')

#Find all links
links = soup.find_all('a', href=True)
for link in links:
    print(link['href'])

soup.find('p', id='first')
soup.find_all('p', class_='paragraph')
soup.find_all('img', src=re.compile('\.gif$'))

#Insert new tags
h2 = soup.new_tag('h2')
h2.string = 'This is a second-level header'
soup.find('p', id='first').insert(0,h2) #Insert before the <p> tag
soup.find('p', id='first').insert(1,h2) #Insert after the <p> tag
soup.find('p', id='first').append(soup.new_tag('h1')) #Insert after the <p> tag

#Add attribute to a tag
soup.head['style'] = 'bold'

#Delete tags adn attributes
print(soup.title.extract()) #Removes the tag and returns it
print(soup.title.extract()) #Removes the tag and returns nothing

#Find comments
from bs4 import Comments
for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
    print(comment)

#Get HTML text
str(soup.prettify())
print(soup.prettify())

#Parse only a specific part of the document
strainer = SoupStrainer(name='ul', attrs={'class: productLister gridView'})
soup = BeautifulSoup(content, 'html.parser', parse_only=strainer)