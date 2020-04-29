import lxml
import requests
from lxml import etree
from lxml.html import fromstring

with open('text5.html', 'w', encoding="utf-8") as outputFile:
    url = "https://www.fontanka.ru/2020/04/03/69069817/"
    r = requests.get(url.strip())
    html = lxml.html.fromstring(r.text)
    text_block = html.find_class("D5cr")
    for t in text_block:
        for tt in t:
            if tt.tag != 'p': continue
            print(tt.text_content(), file=outputFile, end="\n\n")
    print("rubrics: ", [elm.text_content() for elm in html.find_class("GDhr")])
