import lxml
import requests
from lxml.html import fromstring

with open('text2.html', 'w', encoding="utf-8") as outputFile:
    url = "https://lenta.ru/news/2013/02/19/htcone/"
    r = requests.get(url.strip())
    html = lxml.html.fromstring(r.text)
    topc_text = html.find_class("js-topic__text")
    for t in topc_text[0]:
        if t.tag != 'p': continue
        print(t.text_content(), file=outputFile, end="\n\n")
        # print(etree.tounicode(t))
