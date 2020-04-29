import matplotlib.pyplot as plt
import numpy as np
from lxml import etree
from tqdm import tqdm

LDA_MODULE_NAME = 'lda_model_full2'
GEN_CHART_NAME = 'GenerateDistributionOfParagraphWordCountsChart'
PATH_TO_PARAPHRASES = r'C:\Users\kiva0319\PycharmProjects\Diploma2020\processed\paraphrases.xml'

data_words = []
bad_paragraphs = 0

root = etree.parse(PATH_TO_PARAPHRASES)
root = root.getroot()


def extact_paragraphs(element_paragraphs_1):
    bad_paragraphs = 0
    for paragraph in element_paragraphs_1:
        if int(paragraph.attrib.get("words")) >= 5:
            words = []
            for word in paragraph.text.split(";"):
                if word.isalpha():
                    words.append(word)
            data_words.append(words)
        else:
            bad_paragraphs += 1
    return bad_paragraphs


for element in tqdm(root[1]):
    element_paragraphs_1 = element[14]
    element_paragraphs_2 = element[15]
    bad_paragraphs += extact_paragraphs(element_paragraphs_1)
    bad_paragraphs += extact_paragraphs(element_paragraphs_2)

print("Number of th bad paragraphs:", bad_paragraphs)

doc_lens = [len(d) for d in data_words]

plt.figure(figsize=(16, 7), dpi=160)
plt.hist(doc_lens, bins=3000, color='navy')

plt.text(90, 2800, "Mean   : " + str(round(np.mean(doc_lens))), fontdict=dict(size=30))
plt.text(90, 2300, "Median : " + str(round(np.median(doc_lens))), fontdict=dict(size=30))
plt.text(90, 1800, "Stdev   : " + str(round(np.std(doc_lens))), fontdict=dict(size=30))
plt.text(90, 1300, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))), fontdict=dict(size=30))
plt.text(90, 800, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))), fontdict=dict(size=30))

plt.gca().set(xlim=(0, 125), ylabel='Number of Paragraphs', xlabel='Paragraph Word Count')
plt.tick_params(size=16)
plt.xticks(np.linspace(0, 125, 9))
plt.title('Distribution of Paragraph Word Counts', fontdict=dict(size=22))
plt.savefig(GEN_CHART_NAME)

print("Image saved successfully!")
plt.show()
