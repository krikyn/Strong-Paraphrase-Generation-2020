import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download("stopwords")

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from stop_words import get_stop_words


def remove_duplicates(x):
    return list(dict.fromkeys(x))


def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords_nltk \
              and token != " " \
              and token.strip() not in punctuation]
    return tokens


mystem = Mystem()
russian_stopwords_nltk = stopwords.words("russian")
stop_words_extralib = list(get_stop_words('ru'))
stop_words_extralib = remove_duplicates(
    [x for x in mystem.lemmatize(" ".join(stop_words_extralib)) if x not in [" ", "\n"] and x not in punctuation])
russian_stopwords_nltk.extend(stop_words_extralib)
print("Length stopwords list:", len(russian_stopwords_nltk))
print(preprocess_text("Ну что сказать, я вижу кто-то наступил на грабли, Ты разочаровал меня, ты был натравлен."))
