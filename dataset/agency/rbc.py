import lxml
import requests
from lxml.html import fromstring

with open('text6.html', 'w', encoding="utf-8") as outputFile:
    url = "http://web.archive.org/web/" + "https://www.rbc.ru/society/08/05/2015/554ce3039a79477fcaadf9ee"
    # url = "http://web.archive.org/web/20130325085816/http://top.rbc.ru/society/01/01/2013/839227.shtml"
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
    r = requests.get(url, cookies="", headers={'User-Agent': user_agent})
    html = lxml.html.fromstring(r.text)
    topc_text = html.find_class("_ga1_on_ visible")
    if topc_text == []:
        # print(html.find_class("article__text__overview")[0].text_content(), file=outputFile, end="\n\n")
        topc_text = html.find_class("article__text")
        print("rubrics: ", [elm.text.strip() for elm in html.find_class("article__tags__link")])
    for t in topc_text[0]:
        if t.tag != 'p': continue
        print(t.text_content(), file=outputFile, end="\n\n")
        # print(etree.tounicode(t))
