from lxml import etree
from tqdm import tqdm

NEWLINE_TOKEN = "<|n|>"
END_TOKEN = "<|endoftext|>"
PARAPHRASE_TOKEN = " ПРФРЗ: "
TEXT_FRAGMENT_END = "\n" + END_TOKEN + "\n"

root = etree.parse(
    r'C:\Users\kiva0319\IdeaProjects\hrdmd1803\Strong-Paraphrase-Generation-2020\raw_data\paraphrases.xml')
root = root.getroot()
corpus = etree.SubElement(root, "corpus")

non_negative_class_count = 0

with open(
        "C:/Users/kiva0319/IdeaProjects/hrdmd1803/Strong-Paraphrase-Generation-2020/processed/for_train"
        "/title_paraphrase_marked.txt",
        'w', encoding="utf-8") as outputFile:
    for element in tqdm(root[1]):
        id = element[0].text
        id_1 = element[1].text
        id_2 = element[2].text
        title_1 = element[3].text
        title_2 = element[4].text
        jaccard = element[5].text
        clas = element[6].text

        if clas != '-1':
            non_negative_class_count += 1
            print(title_1 + PARAPHRASE_TOKEN + title_2, file=outputFile, end=TEXT_FRAGMENT_END)

print("non_negative_class_count =", non_negative_class_count)
