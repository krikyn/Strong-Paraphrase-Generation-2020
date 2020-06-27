import collections
import os
import re

import numpy as np
from six.moves import cPickle

from .functional import *


class BatchLoader:
    def __init__(self, data_files, idx_files, tensor_files, path='../../'):
        self.data_files = data_files
        self.idx_files = idx_files
        self.tensor_files = tensor_files

        self.blind_symbol = ''
        self.pad_token = '_'
        self.go_token = '>'
        self.end_token = '|'
        self.a_token = '?'

        idx_exists = fold(f_and,
                          [os.path.exists(file) for file in self.idx_files],
                          True)

        tensors_exists = fold(f_and,
                              [os.path.exists(file) for target in self.tensor_files
                               for file in target],
                              True)

        if idx_exists and tensors_exists:
            self.load_preprocessed(self.data_files,
                                   self.idx_files,
                                   self.tensor_files)
            print('preprocessed data was found and loaded')
        else:
            self.preprocess(self.data_files,
                            self.idx_files,
                            self.tensor_files)
            print('data have preprocessed')

        self.word_embedding_index = 0

    def clean_whole_data(self, string):
        string = re.sub(r'^[\d\:]+ ', '', string, 0, re.M)
        string = re.sub(r'\n\s{11}', ' ', string, 0, re.M)
        string = re.sub(r'\n{2}', '\n', string, 0, re.M)

        return string.lower()

    def clean_str(self, string):

        string = re.sub(r"[^가-힣A-Za-z0-9(),!?:;.\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r":", " : ", string)
        string = re.sub(r";", " ; ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def build_character_vocab(self, data):
        chars = list(set(data)) + [self.blind_symbol, self.pad_token, self.go_token, self.end_token]
        chars_vocab_size = len(chars)
        idx_to_char = chars
        char_to_idx = {x: i for i, x in enumerate(idx_to_char)}

        return chars_vocab_size, idx_to_char, char_to_idx

    def build_word_vocab(self, sentences):
        word_counts = collections.Counter(sentences)
        idx_to_word = [x[0] for x in word_counts.most_common()]
        idx_to_word = list(sorted(idx_to_word)) + [self.pad_token, self.go_token, self.end_token]

        words_vocab_size = len(idx_to_word)
        word_to_idx = {x: i for i, x in enumerate(idx_to_word)}

        return words_vocab_size, idx_to_word, word_to_idx

    def preprocess(self, data_files, idx_files, tensor_files):

        data = [open(file, "r").read() for file in data_files]
        merged_data = data[0] + '\n' + data[1]

        self.chars_vocab_size, self.idx_to_char, self.char_to_idx = self.build_character_vocab(merged_data)

        with open(idx_files[1], 'wb') as f:
            cPickle.dump(self.idx_to_char, f)

        data_words = [[line.split() for line in target.split('\n')] for target in data]
        merged_data_words = merged_data.split()

        self.words_vocab_size, self.idx_to_word, self.word_to_idx = self.build_word_vocab(merged_data_words)
        self.max_word_len = np.amax([len(word) for word in self.idx_to_word])
        self.max_seq_len = np.amax([len(line) for target in data_words for line in target])
        self.num_lines = [len(target) for target in data_words]

        with open(idx_files[0], 'wb') as f:
            cPickle.dump(self.idx_to_word, f)

        self.word_tensor = np.array(
            [[list(map(self.word_to_idx.get, line)) for line in target] for target in data_words])
        print(self.word_tensor.shape)
        for i, path in enumerate(tensor_files[0]):
            np.save(path, self.word_tensor[i])

        self.character_tensor = np.array(
            [[list(map(self.encode_characters, line)) for line in target] for target in data_words])
        for i, path in enumerate(tensor_files[1]):
            np.save(path, self.character_tensor[i])

        self.just_words = [word for line in self.word_tensor[0] for word in line]

    def load_preprocessed(self, data_files, idx_files, tensor_files):

        data = [open(file, "r", encoding='utf-8').read() for file in data_files]
        data_words = [[line.split() for line in target.split('\n')] for target in data]
        self.max_seq_len = np.amax([len(line) for target in data_words for line in target])
        self.num_lines = [len(target) for target in data_words]

        [self.idx_to_word, self.idx_to_char] = [cPickle.load(open(file, "rb")) for file in idx_files]

        [self.words_vocab_size, self.chars_vocab_size] = [len(idx) for idx in [self.idx_to_word, self.idx_to_char]]

        [self.word_to_idx, self.char_to_idx] = [dict(zip(idx, range(len(idx)))) for idx in
                                                [self.idx_to_word, self.idx_to_char]]

        self.max_word_len = np.amax([len(word) for word in self.idx_to_word])

        [self.word_tensor, self.character_tensor] = [
            np.array([np.load(target, allow_pickle=True) for target in input_type])
            for input_type in tensor_files]

        self.just_words = [word for line in self.word_tensor[0] for word in line]

    def next_batch(self, batch_size, target_str, start_index):
        target = 0

        indexes = np.array(range(start_index, start_index + batch_size))

        encoder_word_input = [self.word_tensor[target][index] for index in indexes]

        encoder_character_input = [self.character_tensor[target][index] for index in indexes]
        input_seq_len = [len(line) for line in encoder_word_input]
        max_input_seq_len = np.amax(input_seq_len)

        encoded_words = [[idx for idx in line] for line in encoder_word_input]
        decoder_word_input = [[self.word_to_idx[self.go_token]] + line for line in encoder_word_input]
        decoder_character_input = [[self.encode_characters(self.go_token)] + line for line in encoder_character_input]
        decoder_output = [line + [self.word_to_idx[self.end_token]] for line in encoded_words]

        for i, line in enumerate(decoder_word_input):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            decoder_word_input[i] = line + [self.word_to_idx[self.pad_token]] * to_add

        for i, line in enumerate(decoder_character_input):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            decoder_character_input[i] = line + [self.encode_characters(self.pad_token)] * to_add

        for i, line in enumerate(decoder_output):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            decoder_output[i] = line + [self.word_to_idx[self.pad_token]] * to_add

        for i, line in enumerate(encoder_word_input):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            encoder_word_input[i] = [self.word_to_idx[self.pad_token]] * to_add + line[::-1]

        for i, line in enumerate(encoder_character_input):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            encoder_character_input[i] = [self.encode_characters(self.pad_token)] * to_add + line[::-1]

        return np.array(encoder_word_input), np.array(encoder_character_input), \
               np.array(decoder_word_input), np.array(decoder_character_input), np.array(decoder_output)

    def next_embedding_seq(self, seq_len):

        words_len = len(self.just_words)
        seq = [self.just_words[i % words_len]
               for i in np.arange(self.word_embedding_index, self.word_embedding_index + seq_len)]

        result = []
        for i in range(seq_len - 2):
            result.append([seq[i + 1], seq[i]])
            result.append([seq[i + 1], seq[i + 2]])

        self.word_embedding_index = (self.word_embedding_index + seq_len) % words_len - 2

        result = np.array(result)
        return result[:, 0], result[:, 1]

    def go_input(self, batch_size):
        go_word_input = [[self.word_to_idx[self.go_token]] for _ in range(batch_size)]
        go_character_input = [[self.encode_characters(self.go_token)] for _ in range(batch_size)]

        return np.array(go_word_input), np.array(go_character_input)

    def encode_word(self, idx):
        result = np.zeros(self.words_vocab_size)
        result[idx] = 1
        return result

    def decode_word(self, word_idx):
        word = self.idx_to_word[word_idx]
        return word

    def sample_word_from_distribution(self, distribution):
        ix = np.random.choice(range(self.words_vocab_size), p=distribution.ravel())
        x = np.zeros((self.words_vocab_size, 1))
        x[ix] = 1
        return self.idx_to_word[np.argmax(x)]

    def encode_characters(self, characters):
        word_len = len(characters)
        to_add = self.max_word_len - word_len
        characters_idx = [self.char_to_idx[i] for i in characters] + to_add * [self.char_to_idx['']]
        return characters_idx

    def decode_characters(self, characters_idx):
        characters = [self.idx_to_char[i] for i in characters_idx]
        return ''.join(characters)
