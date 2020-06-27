import argparse
import os
import string
import time
from string import punctuation

import torch as t
from model.rvae import RVAE
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu
from pymystem3 import Mystem
from rouge import Rouge
from stop_words import get_stop_words
from torch.autograd import Variable
from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from utils.tensor import preprocess_data


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
    # length = len(lemmas)
    return " ".join(lemmas)


def remove_duplicates(x):
    return list(dict.fromkeys(x))


if __name__ == '__main__':

    assert os.path.exists('./trained_RVAE'), \
        'trained model not found'

    parser = argparse.ArgumentParser(description='Sampler')
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA')
    parser.add_argument('--num-sample', type=int, default=5, metavar='NS')
    parser.add_argument('--num-sentence', type=int, default=10, metavar='NS')
    parser.add_argument('--beam-top', type=int, default=3, metavar='NS',
                        help='beam top (default: 1)')
    parser.add_argument('--beam-size', type=int, default=10, metavar='NS',
                        help='beam size (default: 10)')
    parser.add_argument('--use-file', type=bool, default=True, metavar='NS',
                        help='use file (default: False)')
    parser.add_argument('--test-file', type=str, default='data/test.txt', metavar='NS')
    parser.add_argument('--save-model', type=str, default='./trained_RVAE', metavar='NS',
                        help='trained model save path (default: ./trained_RVAE)')
    args = parser.parse_args()

    if os.path.exists('data/test_word_tensor.npy'):
        os.remove('data/test_word_tensor.npy')
    if os.path.exists('data/test_character_tensor.npy'):
        os.remove('data/test_character_tensor.npy')

    str = ''
    if not args.use_file:
        str = input("Input: ")
    else:
        file_1 = open(args.test_file, 'r')
        data = file_1.readlines()

    data_files = [args.test_file]

    idx_files = ['data/words_vocab.pkl',
                 'data/characters_vocab.pkl']

    tensor_files = [['data/test_word_tensor.npy'],
                    ['data/test_character_tensor.npy']]

    preprocess_data(data_files, idx_files, tensor_files, args.use_file, str)

    batch_loader = BatchLoader(data_files, idx_files, tensor_files)
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    data_files = ['data/super/train_2.txt']

    idx_files = ['data/super/words_vocab_2.pkl',
                 'data/super/characters_vocab_2.pkl']

    tensor_files = [['data/super/train_word_tensor_2.npy'],
                    ['data/super/train_character_tensor_2.npy']]
    batch_loader_2 = BatchLoader(data_files, idx_files, tensor_files)
    parameters_2 = Parameters(batch_loader_2.max_word_len,
                              batch_loader_2.max_seq_len,
                              batch_loader_2.words_vocab_size,
                              batch_loader_2.chars_vocab_size)
    start_time = time.time()
    rvae = RVAE(parameters, parameters_2)
    rvae.load_state_dict(t.load(args.save_model))
    if args.use_cuda:
        rvae = rvae.cuda()
    loading_time = time.time() - start_time
    n_best = args.beam_top
    beam_size = args.beam_size

    assert n_best <= beam_size
    use_cuda = args.use_cuda

    if args.use_file:
        num_sentence = args.num_sentence
    else:
        num_sentence = 1

    mystem = Mystem()
    russian_stopwords = stopwords.words("russian")
    stop_words_extralib = list(get_stop_words('ru'))
    stop_words_extralib = remove_duplicates(
        [x for x in mystem.lemmatize(" ".join(stop_words_extralib)) if x not in [" ", "\n"] and x not in punctuation])
    russian_stopwords.extend(stop_words_extralib)
    scores = {
        "ROUGE 1": [],
        "ROUGE 2": [],
        "ROUGE L": [],
        "BLEU 1": []
    }
    DELETE_STOP_WORDS = True
    DELETECHARS = ''.join([string.punctuation, string.whitespace, "\n", "\xa0", "—", "-"])

    for i in range(len(data)):
        for iteration in range(args.num_sample):
            original = ""
            if args.use_file:
                original = data[i]
            else:
                original = str

            seed = Variable(t.randn([1, parameters.latent_variable_size]))
            seed = seed.cuda()
            results, scores = rvae.sampler(batch_loader, batch_loader_2, 50, seed, args.use_cuda, i, beam_size, n_best)
            best = " ".join([batch_loader_2.decode_word(x[0]) for x in results[0]])
            if batch_loader.end_token in best:
                best = best[:best.index(batch_loader.end_token)]

            text_1_space = original
            text_2_space = best
            text_1_space_1 = get_lemmas_list(original)
            text_2_space_1 = get_lemmas_list(best)
            blue_1 = sentence_bleu([text_1_space_1.split(" ")], text_2_space_1.split(" "), weights=(1, 0, 0, 0))
            rouge = Rouge()
            article_score = rouge.get_scores(text_1_space, text_2_space)[0]
            article_score_1 = rouge.get_scores(text_1_space_1, text_2_space_1)[0]
            scores["ROUGE 1"].append(round(article_score_1['rouge-1']['f'], 4))
            scores["ROUGE 2"].append(round(article_score_1['rouge-2']['f'], 4))
            scores["ROUGE L"].append(round(article_score_1['rouge-l']['f'], 4))
            scores["BLEU 1"].append(round(blue_1, 4))
        print('\n')

for score_name in scores:
    print("_________________")
    print(score_name)
    print(sum(scores[score_name]) / max(len(scores[score_name]), 1))
