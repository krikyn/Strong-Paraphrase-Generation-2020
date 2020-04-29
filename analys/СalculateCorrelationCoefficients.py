from lxml import etree
from tqdm import tqdm

inter_1_scores = []
inter_2_scores = []
inter_3_scores = []
classes = []

root = etree.parse(r'C:\Users\kiva0319\PycharmProjects\Diploma2020\processed\topics.xml')
root = root.getroot()
for element in tqdm(root[1]):
    id = element[0].text
    old_id = element[1].text
    id_1 = element[2].text
    id_2 = element[3].text

    num_of_paragraphs_1 = int(element[10].text)
    num_of_paragraphs_2 = int(element[11].text)

    intersection_1_element = element[20]
    intersection_2_element = element[21]
    intersection_3_element = element[22]
    clas = int(element[23].text)

    inter_1_scores.append(int(intersection_1_element.attrib.get("size")))
    inter_2_scores.append(int(intersection_2_element.attrib.get("size")))
    inter_3_scores.append(int(intersection_3_element.attrib.get("size")))
    if clas == -1:
        classes.append(0)
    else:
        classes.append(1)

meteor_title_scores = []
meteor_article_scores = []
BLUE_w1_titles_scores = []
classes_2 = []

root = etree.parse(r'C:\Users\kiva0319\PycharmProjects\Diploma2020\processed\metrics.xml')
root = root.getroot()
for element in tqdm(root[1]):
    id = element[0].text
    old_id = element[1].text
    id_1 = element[2].text
    id_2 = element[3].text

    clas = int(element[7].text)

    meteor_title = float(element.find("meteor_title").text)
    meteor_article = float(element.find("meteor_article").text)
    BLUE_w1_titles = float(element.find("BLUE_w1_titles").text)

    meteor_title_scores.append(meteor_title)
    meteor_article_scores.append(meteor_article)
    BLUE_w1_titles_scores.append(BLUE_w1_titles)

    if clas == -1:
        classes_2.append(0)
    else:
        classes_2.append(1)

from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot

print(len(inter_1_scores))
print(len(inter_2_scores))
print(len(inter_3_scores))
print(len(classes))

print(inter_2_scores)
print(classes)

print(len(meteor_title_scores))
print(len(meteor_article_scores))
print(len(BLUE_w1_titles_scores))
print(len(classes_2))
# x = inter_1_scores
# pyplot.hist(x, bins=12)
# pyplot.show()

# pyplot.scatter(inter_3_scores, classes)
# # show line plot
# pyplot.show()

from scipy.stats import linregress
from scipy.stats import spearmanr
from scipy.stats import pearsonr

print(linregress(inter_1_scores, classes))
print(linregress(inter_2_scores, classes))
print(linregress(inter_3_scores, classes))
print(linregress(classes, classes))
print()
print()
print(linregress(meteor_title_scores, classes_2))
print(linregress(meteor_article_scores, classes_2))
print(linregress(BLUE_w1_titles_scores, classes_2))
print(linregress(classes_2, classes_2))
print()
print()
print(pearsonr(classes, inter_1_scores))
print(pearsonr(classes, inter_2_scores))
print(pearsonr(classes, inter_3_scores))
print(pearsonr(classes, classes))
print()
print()
print(spearmanr(inter_1_scores, classes))
print(spearmanr(inter_2_scores, classes))
print(spearmanr(inter_3_scores, classes))
print(spearmanr(classes, classes))
print()
print()
print(spearmanr(meteor_title_scores, classes_2))
print(spearmanr(meteor_title_scores, classes_2))
print(spearmanr(BLUE_w1_titles_scores, classes_2))
print(spearmanr(classes_2, classes_2))

import numpy

print(numpy.corrcoef(inter_1_scores, classes)[0, 1])
print(numpy.corrcoef(inter_2_scores, classes)[0, 1])
print(numpy.corrcoef(inter_3_scores, classes)[0, 1])
print(numpy.corrcoef(classes, classes)[0, 1])
