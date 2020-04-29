import lxml
import requests
from lxml.html import fromstring

with open('text4.html', 'w', encoding="utf-8") as outputFile:
    url = "https://kommersant.ru/doc/2135152"
    r = requests.get(url.strip())
    html = lxml.html.fromstring(r.text)
    text_block = html.find_class("article_text_wrapper")
    for t in text_block[0][1:]:
        if t.tag != 'p': continue
        print(t.text_content(), file=outputFile, end="\n\n")
        # print(etree.tounicode(t))
    print("rubrics: ", [elm.text_content() for elm in html.find_class("doc_footer__subs_link")])