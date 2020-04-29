import requests
import lxml
from lxml import etree
from lxml.html import fromstring

root = etree.parse(r'C:\Users\Kiril\PycharmProjects\Diploma2020\data\paraphrases.xml')
root = root.getroot()
corpus = etree.SubElement(root, "corpus")

download_version = "v1"
count_good = 0
count_bad = 0
classes_count = dict()
numbers = set()

bad_pairs = []

words_num = 0
paragraphs_num = 0

words_diff = 0
paragraph_diff = 0

for element in root[1]:
    id = element[0].text
    id_1 = element[1].text
    id_2 = element[2].text
    title_1 = element[3].text
    title_2 = element[4].text
    jaccard = element[5].text
    clas = element[6].text

    text_1 = "none"
    text_2 = "none"

    with open("download/" + download_version + "/" + id_1 + ".txt", 'r', encoding="utf-8") as file:
        text_1 = file.read()

    with open("download/" + download_version + "/" + id_2 + ".txt", 'r', encoding="utf-8") as file:
        text_2 = file.read()

    if len(text_1) < 10 and len(text_2) < 10:
        count_bad += 1
        print("[" + str(count_bad) + "]: id =", id, " spoiled:", id_1, id_2)
        continue

    text_1 = text_1.strip()
    text_2 = text_2.strip()

    words_num_1 = len(text_1.split(" "))
    words_num_2 = len(text_2.split(" "))

    paragraphs_num_1 = len(text_1.split("\n\n"))
    paragraphs_num_2 = len(text_2.split("\n\n"))

    words_num += words_num_1
    words_num += words_num_2
    paragraphs_num += paragraphs_num_1
    paragraphs_num += paragraphs_num_2

    words_diff += abs(words_num_1 - words_num_2)
    paragraph_diff += abs(paragraphs_num_1 - paragraphs_num_2)

    count_good += 1
    print("[" + str(count_good) + "]: id =", id, " paragraphs:", paragraphs_num_1, paragraphs_num_2)

    classes_count[clas] = classes_count.get(clas, 0) + 1
    numbers.add(element[1].text)
    numbers.add(element[2].text)
# print(doc.find_class("js-topic__text"))
# print()

print("Bad: ", count_bad)
print("Good: ", count_good)

print("Words number: ", words_num)
print("Paragraphs number: ", paragraphs_num)

good_text_number = 2.0 * count_good

average_words_number = float(words_num) / float(good_text_number)
average_paragraphs_number = float(paragraphs_num) / float(good_text_number)
print("Average words number: ", average_words_number)
print("Average paragraphs number: ", average_paragraphs_number)

print("Average paragraph size: ", average_words_number / average_paragraphs_number, "words")

print("Average words diff: ", float(words_diff) / float(count_good))
print("Average paragraphs diff: ", float(paragraph_diff) / float(count_good))

print("Classes count: ")
for i in classes_count:
    print("'", i, "' — ", classes_count[i])
