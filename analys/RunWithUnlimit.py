import sys
import threading
from datetime import date

import textstat
from lxml import etree
from nltk.stem.snowball import SnowballStemmer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from rouge import Rouge
from wiki_ru_wordnet import WikiWordnet

threading.stack_size(214748254)
print(threading.stack_size(0))


class Calculation:
    def __call__(self):
        root = etree.parse(r'C:\Users\kiva0319\PycharmProjects\Diploma2020\processed\paraphrases.xml')
        root = root.getroot()
        corpus = etree.SubElement(root, "corpus")

        result_xml = etree.Element('raw_data')
        result_doc = etree.ElementTree(result_xml)

        corpus_info = etree.SubElement(result_xml, 'head')
        etree.SubElement(corpus_info, 'description').text = "—"
        etree.SubElement(corpus_info, 'date').text = str(date.today())
        articles_list = etree.SubElement(result_xml, 'corpus')

        count = 0

        for element in root[1]:
            id = element[0].text
            old_id = element[1].text
            id_1 = element[2].text
            id_2 = element[3].text
            title_1 = element[4].text
            title_2 = element[5].text
            text_1 = element[6].text
            text_2 = element[7].text
            words_title_1 = int(element[8].text)
            words_title_2 = int(element[9].text)
            words_article_1 = int(element[10].text)
            words_article_2 = int(element[11].text)
            num_of_paragraphs_1 = int(element[12].text)
            num_of_paragraphs_2 = int(element[13].text)
            element_paragraphs_1 = element[14].text
            element_paragraphs_2 = element[15].text
            jaccard = element[16].text
            clas = element[17].text

            print(count,id, flush=True)

            # words_max = max(words_max, words_article_1)
            # words_max = max(words_max, words_article_2)
            # chars_max = max(chars_max, len(text_1))
            # chars_max = max(chars_max, len(text_2))
            # continue

            paraphrase = etree.SubElement(articles_list, 'paraphrase')
            etree.SubElement(paraphrase, 'value', name="id").text = id
            etree.SubElement(paraphrase, 'value', name="old_id").text = old_id
            etree.SubElement(paraphrase, 'value', name="id_1").text = id_1
            etree.SubElement(paraphrase, 'value', name="id_2").text = id_2
            etree.SubElement(paraphrase, 'value', name="title_1").text = title_1
            etree.SubElement(paraphrase, 'value', name="title_2").text = title_2
            etree.SubElement(paraphrase, 'value', name="jaccard").text = jaccard
            etree.SubElement(paraphrase, 'value', name="class").text = clas

            # words and paragraphs diff
            etree.SubElement(paraphrase, 'words_title_diff').text = str(abs(words_title_1 - words_title_2))
            etree.SubElement(paraphrase, 'words_article_diff').text = str(abs(words_article_1 - words_article_2))
            etree.SubElement(paraphrase, 'paragraphs_diff').text = str(abs(num_of_paragraphs_1 - num_of_paragraphs_2))

            # flesch_reading_ease
            textstat.textstat.set_lang("ru")
            etree.SubElement(paraphrase, 'flesch_reading_ease_title_1').text = str(
                textstat.flesch_reading_ease(" ".join(title_1.split(";"))))
            etree.SubElement(paraphrase, 'flesch_reading_ease__title_2').text = str(
                textstat.flesch_reading_ease(" ".join(title_2.split(";"))))
            etree.SubElement(paraphrase, 'flesch_reading_ease_article_1').text = str(
                textstat.flesch_reading_ease(" ".join(text_1.split(";"))) / num_of_paragraphs_1)
            etree.SubElement(paraphrase, 'flesch_reading_ease_article_2').text = str(
                textstat.flesch_reading_ease(" ".join(text_2.split(";"))) / num_of_paragraphs_2)

            # BLUE
            weights1 = (1, 0, 0, 0)
            weights2 = (0.5, 0.5, 0, 0)
            weights3 = (0.33, 0.33, 0.33, 0)
            weights4 = (0.25, 0.25, 0.25, 0.25)

            list_title_1 = title_1.split(";")
            list_title_2 = title_2.split(";")
            list_text_1 = text_1.split(";")
            list_text_2 = text_2.split(";")

            etree.SubElement(paraphrase, 'BLUE_w1_titles').text = str(
                sentence_bleu([list_title_1], list_title_2, weights=weights1))
            etree.SubElement(paraphrase, 'BLUE_w2_titles').text = str(
                sentence_bleu([list_title_1], list_title_2, weights=weights2))
            etree.SubElement(paraphrase, 'BLUE_w3_titles').text = str(
                sentence_bleu([list_title_1], list_title_2, weights=weights3))
            etree.SubElement(paraphrase, 'BLUE_w4_titles').text = str(
                sentence_bleu([list_title_1], list_title_2, weights=weights4))

            etree.SubElement(paraphrase, 'BLUE_w1_articles').text = str(
                sentence_bleu([list_text_1], list_text_2, weights=weights1))
            etree.SubElement(paraphrase, 'BLUE_w2_articles').text = str(
                sentence_bleu([list_text_1], list_text_2, weights=weights2))
            etree.SubElement(paraphrase, 'BLUE_w3_articles').text = str(
                sentence_bleu([list_text_1], list_text_2, weights=weights3))
            etree.SubElement(paraphrase, 'BLUE_w4_articles').text = str(
                sentence_bleu([list_text_1], list_text_2, weights=weights4))

            # NIST
            nist_1_titles = 0
            nist_1_articles = 0

            nist_2_titles = 0
            nist_2_articles = 0

            nist_3_titles = 0
            nist_3_articles = 0

            try:
                nist_1_titles = sentence_nist([list_title_1], list_title_2, n=1)
            except ZeroDivisionError:
                print("ZeroDivisionError id =", count)

            try:
                nist_1_articles = sentence_nist([list_text_1], list_text_2, n=1)
            except ZeroDivisionError:
                print("ZeroDivisionError id =", count)

            try:
                nist_2_titles = sentence_nist([list_title_1], list_title_2, n=2)
            except ZeroDivisionError:
                print("ZeroDivisionError id =", count)

            try:
                nist_2_articles = sentence_nist([list_text_1], list_text_2, n=2)
            except ZeroDivisionError:
                print("ZeroDivisionError id =", count)

            try:
                nist_3_titles = sentence_nist([list_title_1], list_title_2, n=3)
            except ZeroDivisionError:
                print("ZeroDivisionError id =", count)

            try:
                nist_3_articles = sentence_nist([list_text_1], list_text_2, n=3)
            except ZeroDivisionError:
                print("ZeroDivisionError id =", count)

            etree.SubElement(paraphrase, 'nist_1_titles').text = str(nist_1_titles)
            etree.SubElement(paraphrase, 'nist_1_articles').text = str(nist_1_articles)

            etree.SubElement(paraphrase, 'nist_2_titles').text = str(nist_2_titles)
            etree.SubElement(paraphrase, 'nist_2_articles').text = str(nist_2_articles)

            etree.SubElement(paraphrase, 'nist_3_titles').text = str(nist_3_titles)
            etree.SubElement(paraphrase, 'nist_3_articles').text = str(nist_3_articles)

            etree.SubElement(paraphrase, 'nist_1_diff').text = str(nist_1_titles - nist_1_articles)
            etree.SubElement(paraphrase, 'nist_2_diff').text = str(nist_2_titles - nist_2_articles)
            etree.SubElement(paraphrase, 'nist_3_diff').text = str(nist_3_titles - nist_3_articles)

            # ROUGE
            title_1_space = title_1.replace(";", " ")
            title_2_space = title_2.replace(";", " ")
            text_1_space = text_1.replace(";", " ")
            text_2_space = text_2.replace(";", " ")

            rouge = Rouge()
            title_score = rouge.get_scores(title_1_space, title_2_space)[0]
            article_score = rouge.get_scores(text_1_space, text_2_space)[0]

            etree.SubElement(paraphrase, 'rouge-1_titles').text = str(title_score['rouge-1']['f'])
            etree.SubElement(paraphrase, 'rouge-2_titles').text = str(title_score['rouge-2']['f'])
            etree.SubElement(paraphrase, 'rouge-L_titles').text = str(title_score['rouge-l']['f'])

            etree.SubElement(paraphrase, 'rouge-1_articles').text = str(article_score['rouge-1']['f'])
            etree.SubElement(paraphrase, 'rouge-2_articles').text = str(article_score['rouge-2']['f'])
            etree.SubElement(paraphrase, 'rouge-L_articles').text = str(article_score['rouge-l']['f'])

            # METEOR
            stemmer = SnowballStemmer("russian")
            wikiwordnet = WikiWordnet()
            etree.SubElement(paraphrase, 'meteor_title').text = str(
                meteor_score([title_1_space], title_2_space, stemmer=stemmer, wordnet=wikiwordnet))
            etree.SubElement(paraphrase, 'meteor_article').text = str(
                meteor_score([text_1_space], text_2_space, stemmer=stemmer, wordnet=wikiwordnet))

            count += 1

        outFile = open("processed/metrics.xml", 'wb')
        result_doc.write(outFile, xml_declaration=True, encoding='utf-8', pretty_print=True)


sys.setrecursionlimit(241800010)
threading.stack_size(0x4000000)
t = threading.Thread(target=Calculation())
t.start()
t.join()
