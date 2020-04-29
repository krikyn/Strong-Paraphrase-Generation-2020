import lxml
import requests
from lxml.html import fromstring

with open('text3.html', 'w', encoding="utf-8") as outputFile:
    url = "http://rg.ru/2015/05/08/markin-anons.html"
    r = requests.get(url.strip())
    html = lxml.html.fromstring(r.text)
    text_block = html.find_class("b-material-wrapper__text")
    for t in text_block[0]:
        if t.tag != 'p': continue
        print(t.text_content(), file=outputFile, end="\n\n")
        # print(etree.tounicode(t))
    print("rubrics: ", [elm.text_content() for elm in html.find_class("b-material-wrapper__rubric")[0].find_class("b-link")])