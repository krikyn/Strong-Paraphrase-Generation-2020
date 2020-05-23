from lxml import etree
from tqdm import tqdm

NEWLINE_TOKEN = "<|n|>"
END_TOKEN = "<|endoftext|>"
PARAPHRASE_TOKEN = " ПРФРЗ: "
PARAGRAPH_END = "\n" + NEWLINE_TOKEN + "\n"
ARTICLE_END = "\n" + NEWLINE_TOKEN + "\n" + END_TOKEN + "\n"

root = etree.parse(
    r'C:\Users\kiva0319\IdeaProjects\hrdmd1803\Strong-Paraphrase-Generation-2020\raw_data\paraphrases.xml')
root = root.getroot()

non_negative_class_count = 0

with open(
        "C:/Users/kiva0319/IdeaProjects/hrdmd1803/Strong-Paraphrase-Generation-2020/processed/for_train"
        "/article_paraphrase_marked.txt",
        'w', encoding="utf-8") as outputFile:
    for element in tqdm(root[1]):
        id_1 = element[1].text
        id_2 = element[2].text
        clas = element[6].text
        text_1 = "none"
        text_2 = "none"

        if clas != '-1':
            non_negative_class_count += 1

            with open(
                    "C:/Users/kiva0319/IdeaProjects/hrdmd1803/Strong-Paraphrase-Generation-2020/download/v1/" + id_1 + ".txt",
                    'r', encoding="utf-8") as file:
                text = file.read()
                if len(text) < 50:
                    continue
                text_1 = text

            with open(
                    "C:/Users/kiva0319/IdeaProjects/hrdmd1803/Strong-Paraphrase-Generation-2020/download/v1/" + id_2 + ".txt",
                    'r', encoding="utf-8") as file:
                text = file.read()
                if len(text) < 50:
                    continue
                text_2 = text

            paragraphs_1 = text_1.split("\n\n")
            paragraphs_2 = text_2.split("\n\n")

            num_of_paragraphs_1_count = 0
            num_of_paragraphs_2_count = 0

            print(NEWLINE_TOKEN, file=outputFile, end="")
            print(PARAGRAPH_END.join(list(filter(lambda p: len(p) > 15, paragraphs_1))), file=outputFile,
                  end=PARAPHRASE_TOKEN)
            print(PARAGRAPH_END.join(list(filter(lambda p: len(p) > 15, paragraphs_2))), file=outputFile,
                  end=ARTICLE_END)

print("non_negative_class_count =", non_negative_class_count)
