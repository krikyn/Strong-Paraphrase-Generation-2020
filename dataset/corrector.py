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
count = 0
download_version = "v1"
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
web_archive_page = "http://web.archive.org/web/2013/"

print("Bad files:")

for element in root[1]:
    id = element[0].text
    title = element[1].text
    agency = element[2].text
    link = element[4].text
    date = element[5].text

    with open("download/" + download_version + "/" + id + ".txt", 'r+', encoding="utf-8") as file:
        text = file.read()
        if len(text) < 10:

            print("[" + str(count) + "]: id =", id, ", agency = ", agency, ", link = ", link, " title = ", title)
            print(text)
            count += 1
            source_examples[agency] = element[4].text
            text_num[agency] = text_num.get(agency, 0) + 1

            try:
                rubrics = "none"
                response = requests.get(web_archive_page + link, cookies="", headers={'User-Agent': user_agent})
                html = lxml.html.fromstring(response.text)
                elements_with_article = html.find_class("_ga1_on_ visible")
                if not elements_with_article:
                    print(html.find_class("article__text__overview")[0].text_content(), file=file, end="\n\n")
                    elements_with_article = html.find_class("article__text")
                    rubrics = str([elm.text.strip() for elm in html.find_class("article__tags__link")])
                for block in elements_with_article[0]:
                    if block.tag != 'p':
                        continue
                    print(block.text_content(), file=file, end="\n\n")
                processed_status = "Downloaded, rubrics: " + rubrics
            except Exception as err:
                print(err)

print("Number of text from agency with one example: ")
for i in source_examples:
    print(i, source_examples[i], "size: ", text_num[i])
print()
print("Link status codes:")
for i in status_codes:
    print("'", i, "' — ", status_codes[i])
