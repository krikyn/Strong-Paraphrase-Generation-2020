import string
from datetime import date
from lxml import etree
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from stop_words import get_stop_words


def stem_and_delete_stopwords(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords \
              and token not in DELETECHARS
              and token not in [" "]
              and token.strip() not in punctuation]
    return tokens


def clean_lemma(lemma):
    return lemma.translate({ord(i): None for i in DELETECHARS})


def clean_char(char):
    if char in DELETECHARS:
        return " "
    else:
        return char


def get_lemmas_list(article_text) -> string:
    article_text = article_text.strip()
    article_text = ''.join([clean_char(ch) for ch in article_text])
    article_text = article_text.lower()
    lemmas = stem_and_delete_stopwords(article_text)
    length = len(lemmas)
    return ";".join(lemmas), length


def remove_duplicates(x):
    return list(dict.fromkeys(x))


mystem = Mystem()
russian_stopwords = stopwords.words("russian")
stop_words_extralib = list(get_stop_words('ru'))
stop_words_extralib = remove_duplicates(
    [x for x in mystem.lemmatize(" ".join(stop_words_extralib)) if x not in [" ", "\n"] and x not in punctuation])
russian_stopwords.extend(stop_words_extralib)

DELETE_STOP_WORDS = True
DELETECHARS = ''.join([string.punctuation, string.whitespace, "\n", "\xa0", "—", "-"])

root = etree.parse(r'C:\Users\kiva0319\PycharmProjects\Diploma2020\data\paraphrases.xml')
root = root.getroot()
corpus = etree.SubElement(root, "corpus")

result_xml = etree.Element('raw_data')
result_doc = etree.ElementTree(result_xml)

corpus_info = etree.SubElement(result_xml, 'head')
etree.SubElement(corpus_info, 'description').text = "—"
element_size = etree.SubElement(corpus_info, 'size')
etree.SubElement(corpus_info, 'date').text = str(date.today())

articles_list = etree.SubElement(result_xml, 'corpus')

count = 0

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

    print(count, flush=True)

    with open("download/v1/" + id_1 + ".txt", 'r', encoding="utf-8") as file:
        text = file.read()
        if len(text) < 50:
            print("bad file id =", id_1)
            continue
        text_1 = text

    with open("download/v1/" + id_2 + ".txt", 'r', encoding="utf-8") as file:
        text = file.read()
        if len(text) < 50:
            print("bad file id =", id_2)
            continue
        text_2 = text

    paraphrase = etree.SubElement(articles_list, 'paraphrase')
    etree.SubElement(paraphrase, 'value', name="id").text = str(count)
    etree.SubElement(paraphrase, 'value', name="old_id").text = id
    etree.SubElement(paraphrase, 'value', name="id_1").text = id_1
    etree.SubElement(paraphrase, 'value', name="id_2").text = id_2
    etree.SubElement(paraphrase, 'value', name="title_1").text, words_title_1 = get_lemmas_list(title_1)
    etree.SubElement(paraphrase, 'value', name="title_2").text, words_title_2 = get_lemmas_list(title_2)
    etree.SubElement(paraphrase, 'value', name="text_1").text, words_article_1 = get_lemmas_list(text_1)
    etree.SubElement(paraphrase, 'value', name="text_2").text, words_article_2 = get_lemmas_list(text_2)

    paragraphs_1 = text_1.split("\n\n")
    paragraphs_2 = text_2.split("\n\n")

    etree.SubElement(paraphrase, 'value', name="words_title_1").text = str(words_title_1)
    etree.SubElement(paraphrase, 'value', name="words_title_2").text = str(words_title_2)

    etree.SubElement(paraphrase, 'value', name="words_article_1").text = str(words_article_1)
    etree.SubElement(paraphrase, 'value', name="words_article_2").text = str(words_article_2)

    num_of_paragraphs_1 = etree.SubElement(paraphrase, 'value', name="num_of_paragraphs_1")
    num_of_paragraphs_2 = etree.SubElement(paraphrase, 'value', name="num_of_paragraphs_2")

    element_paragraphs_1 = etree.SubElement(paraphrase, 'value', name="paragraphs_1")
    element_paragraphs_2 = etree.SubElement(paraphrase, 'value', name="paragraphs_2")

    num_of_paragraphs_1_count = 0
    num_of_paragraphs_2_count = 0

    for paragraph in paragraphs_1:
        if len(paragraph) > 15:
            p_text, words_num = get_lemmas_list(paragraph)
            etree.SubElement(element_paragraphs_1, 'paragraph', words=str(words_num)).text = p_text
            num_of_paragraphs_1_count += 1

    for paragraph in paragraphs_2:
        if len(paragraph) > 15:
            p_text, words_num = get_lemmas_list(paragraph)
            etree.SubElement(element_paragraphs_2, 'paragraph', words=str(words_num)).text = p_text
            num_of_paragraphs_2_count += 1

    num_of_paragraphs_1.text = str(num_of_paragraphs_1_count)
    num_of_paragraphs_2.text = str(num_of_paragraphs_2_count)

    etree.SubElement(paraphrase, 'value', name="jaccard").text = jaccard
    etree.SubElement(paraphrase, 'value', name="class").text = clas
    count += 1

outFile = open("processed/paraphrases.xml", 'wb')
result_doc.write(outFile, xml_declaration=True, encoding='utf-8', pretty_print=True)
