import constants
from logger import save_dataframe, read_dataframe

import numpy as np
import pandas as pd

import torch


class Vocabulary:
    def __init__(self):
        self.word2count = {}
        self.word2index = {}
        # TODO: Take these values from constants
        self.index2word = {0: "<sos>", 1: "<eos>", 2: "<unk>"}
        self.__n_words = len(self.index2word)


    def __len__(self):
        return self.__n_words


    def collectWordsFrom(self, sentences):
        for sentence in sentences:
            for word in sentence:
                self.__addWord(word)


    def save(self, fname):
        df = pd.DataFrame(columns=['word', 'index', 'count'])
        df['word'] = list(self.word2index.keys())
        df['index'] = df['word'].apply(lambda word: self.word2index[word])
        df['count'] = df['word'].apply(lambda word: self.word2count[word])
        save_dataframe(df, fname)


    def load(self, fname):
        df = read_dataframe(fname)
        self.word2count = dict(zip(df['word'], df['count']))
        self.word2index = dict(zip(df['word'], df['index']))

        new_index2word = dict(zip(df['index'], df['word']))
        self.index2word = {**new_index2word, **self.index2word}

        self.__n_words = len(self.index2word)


    def __addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.__n_words
            self.word2count[word] = 1
            self.index2word[self.__n_words] = word
            self.__n_words += 1
        else:
            self.word2count[word] += 1


class NumberEncoder:
    """Encodes words as numbers and decodes numbers as words
    using a given vocabulary"""
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary


    def encodeSentence(self, sentence):
        numbers = [self.__encodeWord(word) for word in sentence]
        numbers.append(constants.EOS_TOKEN)
        return numbers


    def decodeSentence(self, numbers):
        # Remove EOS_TOKEN
        numbers = numbers[:-1]
        return [self.__decodeWord(number) for number in numbers]


    def __encodeWord(self, word):
        if word not in self.vocabulary.word2index.keys():
            return constants.UNK_TOKEN
        return self.vocabulary.word2index[word]


    def __decodeWord(self, number):
        return self.vocabulary.index2word[number]


class TensorEncoder:
    def __init__(self, input_vocab, output_vocab):
        self.input_num_encoder = NumberEncoder(input_vocab)
        self.output_num_encoder = NumberEncoder(output_vocab)


    def encode(self, df, input_column, output_columns):
        cols = [input_column] + output_columns

        df.loc[:,input_column] = df[input_column].apply(self.input_num_encoder.encodeSentence)

        for col in output_columns:
            df.loc[:,col] = df[col].apply(self.output_num_encoder.encodeSentence)

        for col in cols:
            df.loc[:,col] = df[col].apply(self.__list_to_tensor)

        return df


    def decode(self, df, input_column, output_columns):
        cols = [input_column] + output_columns

        for col in cols:
            if type(df[col].iloc[0]) == torch.Tensor:
                df[col] = df[col].apply(self.__tensor_to_list)

        df[input_column] = df[input_column].apply(self.input_num_encoder.decodeSentence)

        for col in output_columns:
            df[col] = df[col].apply(self.output_num_encoder.decodeSentence)

        return df


    def __list_to_tensor(self, sentence):
        return torch.tensor(sentence, dtype=torch.long, device=constants.DEVICE).view(-1, 1)


    def __tensor_to_list(self, tensor):
        return tensor.view(-1).tolist()
