from datetime import date

from lxml import etree

root = etree.parse(r'C:\Users\kiva0319\PycharmProjects\Diploma2020\processed\metrics.xml')
root = root.getroot()
corpus = etree.SubElement(root, "corpus")

count = 0

# chars_max = 0
# words_max = 0

BLUE_w1_titles_list = {-1: [], 0: [], 1: []}
BLUE_w1_articles_list = {-1: [], 0: [], 1: []}

scores = {
    "flesch_reading_ease_title_diff": {-1: [], 0: [], 1: []},
    "flesch_reading_ease_article_diff": {-1: [], 0: [], 1: []},
    "BLUE_w1_titles": {-1: [], 0: [], 1: []},
    "BLUE_w2_titles": {-1: [], 0: [], 1: []},
    "BLUE_w3_titles": {-1: [], 0: [], 1: []},
    "BLUE_w4_titles": {-1: [], 0: [], 1: []},
    "BLUE_w1_articles": {-1: [], 0: [], 1: []},
    "BLUE_w2_articles": {-1: [], 0: [], 1: []},
    "BLUE_w3_articles": {-1: [], 0: [], 1: []},
    "BLUE_w4_articles": {-1: [], 0: [], 1: []},
    "nist_1_titles": {-1: [], 0: [], 1: []},
    "nist_1_articles": {-1: [], 0: [], 1: []},
    "nist_2_titles": {-1: [], 0: [], 1: []},
    "nist_2_articles": {-1: [], 0: [], 1: []},
    "nist_3_titles": {-1: [], 0: [], 1: []},
    "nist_3_articles": {-1: [], 0: [], 1: []},
    "rouge_1_titles": {-1: [], 0: [], 1: []},
    "rouge_2_titles": {-1: [], 0: [], 1: []},
    "rouge_L_titles": {-1: [], 0: [], 1: []},
    "rouge_1_articles": {-1: [], 0: [], 1: []},
    "rouge_2_articles": {-1: [], 0: [], 1: []},
    "rouge_L_articles": {-1: [], 0: [], 1: []},
    "meteor_title": {-1: [], 0: [], 1: []},
    "meteor_article": {-1: [], 0: [], 1: []},
}

# for element in [root[1][0], root[1][1]]:
for element in root[1]:
    id = element[0].text
    old_id = element[1].text
    id_1 = element[2].text
    id_2 = element[3].text
    title_1 = element[4].text
    title_2 = element[5].text
    jaccard = element[6].text
    c = int(element[7].text)

    words_title_diff = float(element.find("words_title_diff").text)
    words_article_diff = float(element.find("words_article_diff").text)
    paragraphs_diff = float(element.find("paragraphs_diff").text)

    flesch_reading_ease_title_1 = float(element.find("flesch_reading_ease_title_1").text)
    flesch_reading_ease__title_2 = float(element.find("flesch_reading_ease__title_2").text)
    flesch_reading_ease_article_1 = float(element.find("flesch_reading_ease_article_1").text)
    flesch_reading_ease_article_2 = float(element.find("flesch_reading_ease_article_2").text)

    scores["flesch_reading_ease_title_diff"][c].append(abs(flesch_reading_ease_title_1 - flesch_reading_ease__title_2))
    scores["flesch_reading_ease_article_diff"][c].append(
        abs(flesch_reading_ease_article_1 - flesch_reading_ease_article_2))

    BLUE_w1_titles = float(element.find("BLUE_w1_titles").text)
    BLUE_w2_titles = float(element.find("BLUE_w2_titles").text)
    BLUE_w3_titles = float(element.find("BLUE_w3_titles").text)
    BLUE_w4_titles = float(element.find("BLUE_w4_titles").text)

    scores["BLUE_w1_titles"][c].append(BLUE_w1_titles)
    scores["BLUE_w2_titles"][c].append(BLUE_w2_titles)
    scores["BLUE_w3_titles"][c].append(BLUE_w3_titles)
    scores["BLUE_w4_titles"][c].append(BLUE_w4_titles)

    BLUE_w1_articles = float(element.find("BLUE_w1_articles").text)
    BLUE_w2_articles = float(element.find("BLUE_w2_articles").text)
    BLUE_w3_articles = float(element.find("BLUE_w3_articles").text)
    BLUE_w4_articles = float(element.find("BLUE_w4_articles").text)

    scores["BLUE_w1_articles"][c].append(BLUE_w1_articles)
    scores["BLUE_w2_articles"][c].append(BLUE_w2_articles)
    scores["BLUE_w3_articles"][c].append(BLUE_w3_articles)
    scores["BLUE_w4_articles"][c].append(BLUE_w4_articles)

    nist_1_titles = float(element.find("nist_1_titles").text)
    nist_1_articles = float(element.find("nist_1_articles").text)

    scores["nist_1_titles"][c].append(nist_1_titles)
    scores["nist_1_articles"][c].append(nist_1_articles)

    nist_2_titles = float(element.find("nist_2_titles").text)
    nist_2_articles = float(element.find("nist_2_articles").text)

    scores["nist_2_titles"][c].append(nist_2_titles)
    scores["nist_2_articles"][c].append(nist_2_articles)

    nist_3_titles = float(element.find("nist_3_titles").text)
    nist_3_articles = float(element.find("nist_3_articles").text)

    scores["nist_3_titles"][c].append(nist_3_titles)
    scores["nist_3_articles"][c].append(nist_3_articles)

    nist_1_diff = float(element.find("nist_1_diff").text)
    nist_2_diff = float(element.find("nist_2_diff").text)
    nist_3_diff = float(element.find("nist_3_diff").text)

    rouge_1_titles = float(element.find("rouge-1_titles").text)
    rouge_2_titles = float(element.find("rouge-2_titles").text)
    rouge_L_titles = float(element.find("rouge-L_titles").text)

    scores["rouge_1_titles"][c].append(rouge_1_titles)
    scores["rouge_2_titles"][c].append(rouge_2_titles)
    scores["rouge_L_titles"][c].append(rouge_L_titles)

    rouge_1_articles = float(element.find("rouge-1_articles").text)
    rouge_2_articles = float(element.find("rouge-2_articles").text)
    rouge_L_articles = float(element.find("rouge-L_articles").text)

    scores["rouge_1_articles"][c].append(rouge_1_articles)
    scores["rouge_2_articles"][c].append(rouge_2_articles)
    scores["rouge_L_articles"][c].append(rouge_L_articles)

    meteor_title = float(element.find("meteor_title").text)
    meteor_article = float(element.find("meteor_article").text)

    scores["meteor_title"][c].append(meteor_title)
    scores["meteor_article"][c].append(meteor_article)

# result_xml = etree.Element('raw_data')
# result_doc = etree.ElementTree(result_xml)
#
# corpus_info = etree.SubElement(result_xml, 'head')
# etree.SubElement(corpus_info, 'description').text = "—"
# etree.SubElement(corpus_info, 'date').text = str(date.today())
#
# outFile = open("processed/summary.xml", 'wb')
# result_doc.write(outFile, xml_declaration=True, encoding='utf-8', pretty_print=True)
# print(BLUE_w1_titles_list)
# print(BLUE_w1_articles_list)
# print(len(BLUE_w1_titles_list))
# print(len(BLUE_w1_articles_list))
# print(sum(BLUE_w1_titles_list))
# print(sum(BLUE_w1_articles_list))
# 
# print()
# print()
# print(sum(BLUE_w1_titles_list) / len(BLUE_w1_titles_list), min(BLUE_w1_titles_list), max(BLUE_w1_titles_list))
# print(sum(BLUE_w1_articles_list) / len(BLUE_w1_articles_list), min(BLUE_w1_articles_list), max(BLUE_w1_articles_list))
# 
# BLUE_w1_titles_list.sort()
# BLUE_w1_articles_list.sort()
# print(BLUE_w1_titles_list)
# print(BLUE_w1_articles_list)

# print(sum(BLUE_w1_titles_list[-1]) / len(BLUE_w1_titles_list[-1]), min(BLUE_w1_titles_list[-1]), max(BLUE_w1_titles_list[-1]))
# print(sum(BLUE_w1_articles_list[-1]) / len(BLUE_w1_articles_list[-1]), min(BLUE_w1_articles_list[-1]), max(BLUE_w1_articles_list[-1]))
# print()
# print()
# print(sum(BLUE_w1_titles_list[0]) / len(BLUE_w1_titles_list[0]), min(BLUE_w1_titles_list[0]), max(BLUE_w1_titles_list[0]))
# print(sum(BLUE_w1_articles_list[0]) / len(BLUE_w1_articles_list[0]), min(BLUE_w1_articles_list[0]), max(BLUE_w1_articles_list[0]))
# print()
# print()
# print(sum(BLUE_w1_titles_list[1]) / len(BLUE_w1_titles_list[1]), min(BLUE_w1_titles_list[1]), max(BLUE_w1_titles_list[1]))
# print(sum(BLUE_w1_articles_list[1]) / len(BLUE_w1_articles_list[1]), min(BLUE_w1_articles_list[1]), max(BLUE_w1_articles_list[1]))
# print()
# print()

result_xml = etree.Element('raw_data')
result_doc = etree.ElementTree(result_xml)

corpus_info = etree.SubElement(result_xml, 'head')
etree.SubElement(corpus_info, 'description').text = "—"
etree.SubElement(corpus_info, 'date').text = str(date.today())

for score_name in scores:
    print("_________________")
    print(score_name)
    print("[ -1 ]", sum(scores[score_name][-1]) / len(scores[score_name][-1]))
    print("[  0 ]", sum(scores[score_name][0]) / len(scores[score_name][0]))
    print("[  1 ]", sum(scores[score_name][1]) / len(scores[score_name][1]))
    print("[0&&1]", float(sum(scores[score_name][0]) + sum(scores[score_name][1])) / float(
        len(scores[score_name][0]) + len(scores[score_name][1])))

    articles_list = etree.SubElement(result_xml, score_name)
    etree.SubElement(articles_list, 'negative').text = str(sum(scores[score_name][-1]) / len(scores[score_name][-1]))
    etree.SubElement(articles_list, 'neutral').text = str(sum(scores[score_name][0]) / len(scores[score_name][0]))
    etree.SubElement(articles_list, 'positive').text = str(sum(scores[score_name][1]) / len(scores[score_name][1]))
    etree.SubElement(articles_list, 'not_negative').text = str(
        float(sum(scores[score_name][0]) + sum(scores[score_name][1])) / float(
            len(scores[score_name][0]) + len(scores[score_name][1])))
    # print(len(scores[score_name][0]))
    # print(len(scores[score_name][1]))

outFile = open("processed/summary.xml", 'wb')
result_doc.write(outFile, xml_declaration=True, encoding='utf-8', pretty_print=True)
