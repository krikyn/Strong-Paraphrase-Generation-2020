import lxml
import requests
from lxml.html import fromstring

with open('text5.html', 'w', encoding="utf-8") as outputFile:
    url = "http://ria.ru/world/20150509/1063505600.html"
    r = requests.get(url.strip())
    html = lxml.html.fromstring(r.text)
    print(html.find_class("article__announce-text")[0].text_content(), file=outputFile, end="\n\n")
    text_block = html.find_class("article__text")
    for t in text_block:
        if t.text_content() == "": continue
        print(t.text_content(), file=outputFile, end="\n\n")
        # print(etree.tounicode(t))
    print("rubrics: ", [elm.text_content() for elm in html.find_class("article__tags-item")])
