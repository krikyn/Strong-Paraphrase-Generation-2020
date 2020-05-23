import requests
import lxml
from lxml import etree
from lxml.html import fromstring

root = etree.parse(r'C:\Users\Kiril\PycharmProjects\Diploma2020\data\corpus.xml')
root = root.getroot()
corpus = etree.SubElement(root, "corpus")

source_examples = dict()
text_num = dict()
status_codes = dict()
failed_elements = dict()
count = 1214
download_version = "v1"
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
web_archive_page = "http://web.archive.org/web/"

for element in root[1][1214:]:
    id = element[0].text
    text = element[1].text
    agency = element[2].text
    link = element[4].text
    date = element[5].text

    source_examples[agency] = element[4].text
    text_num[agency] = text_num.get(agency, 0) + 1

    count += 1
    processed_status = "Unknown agency '" + agency + "'"
    print("[" + str(count) + "]: id =", id, ", agency = ", agency, ", link = ", link, ", processed_status =", end=" ")

    try:
        with open("download/" + download_version + "/" + id + ".txt", 'w', encoding="utf-8") as outputFile:
            if agency == "РБК":
                rubrics = "none"
                response = requests.get(web_archive_page + link, cookies="", headers={'User-Agent': user_agent})
                html = lxml.html.fromstring(response.text)
                elements_with_article = html.find_class("_ga1_on_ visible")
                if not elements_with_article:
                    elements_with_article = html.find_class("article__text")
                    rubrics = str([elm.text.strip() for elm in html.find_class("article__tags__link")])
                for block in elements_with_article[0]:
                    if block.tag != 'p':
                        continue
                    print(block.text_content(), file=outputFile, end="\n\n")
                processed_status = "Downloaded, rubrics: " + rubrics
            elif agency == "Лента.ру":
                rubrics = "none"
                response = requests.get(link)
                html = lxml.html.fromstring(response.text)
                elements_with_article = html.find_class("js-topic__text")
                for block in elements_with_article[0]:
                    if block.tag != 'p': continue
                    print(block.text_content(), file=outputFile, end="\n\n")
                processed_status = "Downloaded, rubrics: " + rubrics
            elif agency == "Российская газета":
                response = requests.get(link)
                html = lxml.html.fromstring(response.text)
                text_block = html.find_class("b-material-wrapper__text")
                for block in text_block[0]:
                    if block.tag != 'p': continue
                    print(block.text_content(), file=outputFile, end="\n\n")
                rubrics = str(
                    [elm.text_content() for elm in
                     html.find_class("b-material-wrapper__rubric")[0].find_class("b-link")])
                processed_status = "Downloaded, rubrics: " + rubrics
            elif agency == "КоммерсантЪ":
                response = requests.get(link.strip())
                html = lxml.html.fromstring(response.text)
                text_block = html.find_class("article_text_wrapper")
                for block in text_block[0][1:]:
                    if block.tag != 'p': continue
                    print(block.text_content(), file=outputFile, end="\n\n")
                rubrics = str([elm.text_content() for elm in html.find_class("doc_footer__subs_link")])
                processed_status = "Downloaded, rubrics: " + rubrics
            elif agency == "РИА Новости":
                response = requests.get(link)
                html = lxml.html.fromstring(response.text)
                print(html.find_class("article__announce-text")[0].text_content(), file=outputFile, end="\n\n")
                text_block = html.find_class("article__text")
                for block in text_block:
                    if block.text_content() == "": continue
                    print(block.text_content(), file=outputFile, end="\n\n")
                rubrics = str([elm.text_content() for elm in html.find_class("article__tags-item")])
                processed_status = "Downloaded, rubrics: " + rubrics
            elif agency == "ИноСМИ":
                response = requests.get(link)
                html = lxml.html.fromstring(response.text)
                text = etree.tounicode(html.find_class("article-body")[0][0]).replace("<p>", "").replace("</p>", "")
                for block in text.split("<br/><br/>"):
                    print(block, file=outputFile, end="\n\n")
                rubrics = str([elm.text_content() for elm in html.find_class("article-header__story")])
                processed_status = "Downloaded, rubrics: " + rubrics
            elif agency == "Фонтанка.ру":
                response = requests.get(link)
                html = lxml.html.fromstring(response.text)
                text_block = html.find_class("D5cr")
                for block in text_block:
                    for sub_block in block:
                        if sub_block.tag != 'p': continue
                        print(sub_block.text_content(), file=outputFile, end="\n\n")
                rubrics = str([elm.text_content() for elm in html.find_class("GDhr")])
                processed_status = "Downloaded, rubrics: " + rubrics
            else:
                print("Failed", file=outputFile)
    except Exception as err:
        processed_status = "Failed — " + str(err)
    print(processed_status)

print("Completed!")
print()
print("Number of text from agency with one example: ")
for i in source_examples:
    print(i, source_examples[i], "size: ", text_num[i])
print()
print("Link status codes:")
for i in status_codes:
    print("'", i, "' — ", status_codes[i])
