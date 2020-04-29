import lxml
import requests
from lxml import etree
from lxml.html import fromstring, tostring

with open('text5.html', 'w', encoding="utf-8") as outputFile:
    url = "http://inosmi.ru/russia/20150508/227923804.html"
    r = requests.get(url.strip())
    html = lxml.html.fromstring(r.text)
    text = etree.tounicode(html.find_class("article-body")[0][0]).replace("<p>", "").replace("</p>", "")
    for t in text.split("<br/><br/>"):
        print(t, file=outputFile, end="\n\n")
print("rubrics: ", [elm.text_content() for elm in html.find_class("article-header__story")])
