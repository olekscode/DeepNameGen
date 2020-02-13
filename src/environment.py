import constants
from seq2seq import Seq2Seq
from preprocessing import Vocabulary
from logger import log, save_dataframe, read_dataframe

import os

import numpy as np
import pandas as pd

import torch


class Environment:
    def __init__(self):
        self.train_methods = None
        self.valid_methods = None
        self.test_methods = None
        self.input_vocab = Vocabulary()
        self.output_vocab = Vocabulary()
        self.model = None
        self.history = None
        self.iters_completed = None
        self.total_training_time = None


    def save(self):
        log('Saving the state of the environment')
        self.__save_methods()
        self.__save_vocabs()
        self.__save_model()
        self.__save_history()
        self.__save_iters_completed()
        self.__save_total_training_time()


    def save_data(self):
        log('Saving dataset splits and vocabularies')
        self.__save_methods()
        self.__save_vocabs()


    def save_train(self):
        log('Saving the state of training')
        self.__save_model()
        self.__save_history()
        self.__save_iters_completed()
        self.__save_total_training_time()


    def load(self):
        log('Loading the last saved environment')
        self.__load_methods()
        self.__load_vocabs()
        self.__load_model()
        self.__load_history()
        self.__load_iters_completed()
        self.__load_total_training_time()

        log("Starting from iteration {}\n{} more iterations to go ({:.1f}%)".format(
            self.iters_completed + 1,
            constants.NUM_ITER - self.iters_completed,
            (constants.NUM_ITER - self.iters_completed) / constants.NUM_ITER * 100))


    def initialize_new(self):
        log('Initializing a new environment')
        self.__init_methods()
        self.__init_vocabs()
        self.__init_model()
        self.__init_history()
        self.__init_iters_completed()
        self.__init_total_training_time()


    def __save_methods(self):
        log('Saving train, validation, and test methods')
        join = lambda tokens: ' '.join(tokens)
        
        save_dataframe(self.train_methods.applymap(join), constants.TRAIN_METHODS_FILE)
        save_dataframe(self.valid_methods.applymap(join), constants.VALID_METHODS_FILE)
        save_dataframe(self.test_methods.applymap(join), constants.TEST_METHODS_FILE)


    def __save_vocabs(self):
        log('Saving vocabularies')
        self.input_vocab.save(constants.INPUT_VOCAB_FILE)
        self.output_vocab.save(constants.OUTPUT_VOCAB_FILE)


    def __save_model(self):
        log('Saving the model')
        torch.save(self.model.state_dict(), constants.TRAINED_MODEL_FILE)


    def __save_history(self):
        log('Saving score history')
        save_dataframe(self.history, constants.HISTORIES_FILE)


    def __save_iters_completed(self):
        log('Saving the number of completed iterations')
        with open(constants.ITERS_COMPLETED_FILE, 'w') as f:
            f.write(str(self.iters_completed))


    def __save_total_training_time(self):
        log('Saving the total training time')
        with open(constants.TRAINING_TIME_FILE, 'w') as f:
            f.write(str(self.total_training_time))


    def __load_methods(self):
        log('Loading train, validation, and test methods')
        self.train_methods = read_dataframe(constants.TRAIN_METHODS_FILE)
        self.valid_methods = read_dataframe(constants.VALID_METHODS_FILE)
        self.test_methods = read_dataframe(constants.TEST_METHODS_FILE)

        self.__log_split()
        self.__split_tokens()


    def __load_vocabs(self):
        log('Loading vocabularies')
        self.input_vocab.load(constants.INPUT_VOCAB_FILE)
        self.output_vocab.load(constants.OUTPUT_VOCAB_FILE)
        self.__log_vocabs()


    def __load_model(self):
        self.__init_model()

        log('Loading the last trained model')
        self.model.load_state_dict(torch.load(constants.TRAINED_MODEL_FILE))


    def __load_history(self):
        log('Loading score history')
        self.history = read_dataframe(constants.HISTORIES_FILE)


    def __load_iters_completed(self):
        log('Loading the number of completed iterations')
        with open(constants.ITERS_COMPLETED_FILE, 'r') as f:
            self.iters_completed = int(f.read())


    def __load_total_training_time(self):
        log('Loading the total training time')
        with open(constants.TRAINING_TIME_FILE, 'r') as f:
            self.total_training_time = float(f.read())


    def __init_methods(self):
        self.__load_dataset()
        self.__split_tokens()
        self.__filter_dataset()
        self.__split_dataset()

        self.__log_split()


    def __init_vocabs(self):
        log('Collecting the tokens from train set into the vocabularies')
        self.input_vocab.collectWordsFrom(self.train_methods['source'])
        self.output_vocab.collectWordsFrom(self.train_methods['name'])
        self.__log_vocabs()


    def __init_model(self):
        log('Initializing an empty model')
        self.model = Seq2Seq(
            input_size=len(self.input_vocab),
            output_size=len(self.output_vocab),
            hidden_size=constants.HIDDEN_SIZE,
            learning_rate=constants.LEARNING_RATE,
            teacher_forcing_ratio=constants.TEACHER_FORCING_RATIO,
            device=constants.DEVICE)

        log(str(self.model))


    def __init_history(self):
        log('Initializing an empty DataFrame for score history')
        self.history = pd.DataFrame(
            columns=['Loss', 'BLEU', 'ROUGE', 'F1', 'num_names'])


    def __init_iters_completed(self):
        log('Setting the number of completed iterations equal to 0')
        self.iters_completed = 0


    def __init_total_training_time(self):
        log('Setting the total training time equal to 0')
        self.total_training_time = 0


    def __load_dataset(self):
        log('Loading the dataset')
        self.__methods = read_dataframe(constants.DATASET_FILE)
        self.__methods = self.__methods[['name', 'source']]


    def __split_dataset(self):
        log('Splitting the dataset into train, validation, and test sets')

        if constants.TRAIN_PROP + constants.VALID_PROP + constants.TEST_PROP != 1.0:
            raise ValueError("Train, validation, and test proportions don't sum up to 1.")

        test_size = int(constants.TEST_PROP * len(self.__methods))
        valid_size = int(constants.VALID_PROP * len(self.__methods))
        train_size = int(constants.TRAIN_PROP * len(self.__methods))

        indices = np.random.permutation(len(self.__methods))

        train_idx = indices[:train_size]
        valid_idx = indices[train_size:(train_size + valid_size)]
        test_idx = indices[-test_size:]

        self.train_methods = self.__methods.iloc[train_idx]
        self.valid_methods = self.__methods.iloc[valid_idx]
        self.test_methods = self.__methods.iloc[test_idx]


    def __filter_dataset(self):
        # TODO: Move it elsewhere
        log('Filtering data from dataset')
        initial_size = len(self.__methods)

        log('Removing empty methods')
        self.__methods = self.__methods[self.__methods['source'].apply(
            lambda code: len(code) > 0)]

        log('{} methods left ({:.2f}%)'.format(
            len(self.__methods), len(self.__methods) / initial_size * 100))

        log('Removing methods with empty names')
        self.__methods = self.__methods[self.__methods['name'].apply(
            lambda name: len(name) > 0)]

        log('{} methods left ({:.2f}%)'.format(
            len(self.__methods), len(self.__methods) / initial_size * 100))

        log('Removing getters')
        self.__methods = self.__methods[self.__methods['name'] != self.__methods['source']]

        log('{} methods left ({:.2f}%)'.format(
            len(self.__methods), len(self.__methods) / initial_size * 100))

        log('Removing test methods')
        self.__methods = self.__methods[self.__methods['name'].apply(lambda name: name[0] != 'test')]

        log('{} methods left ({:.2f}%)'.format(
            len(self.__methods), len(self.__methods) / initial_size * 100))

        log('Removing abstract methods')
        self.__methods = self.__methods[self.__methods['source'].apply(lambda source: source != ['self', 'subclass', 'responsibility'])]

        log('{} methods left ({:.2f}%)'.format(
            len(self.__methods), len(self.__methods) / initial_size * 100))

        log('Removing shouldNotImplement methods')
        self.__methods = self.__methods[self.__methods['source'].apply(lambda source: source != ['self', 'should', 'not', 'implement'])]

        log('{} methods left ({:.2f}%)'.format(
            len(self.__methods), len(self.__methods) / initial_size * 100))

        log('Removing keywords')
        self.__methods = self.__methods[self.__methods['source'].apply(lambda source: not set(source).issubset(['self', 'should', 'not', 'implement']))]

        log('{} methods left ({:.2f}%)'.format(
            len(self.__methods), len(self.__methods) / initial_size * 100))

        if constants.DROP_DUPLICATES:
            log('Removing duplicate methods')
            self.__methods = self.__methods.loc[self.__methods.astype(str).drop_duplicates().index]

            log('{} methods left ({:.2f}%)'.format(
                len(self.__methods), len(self.__methods) / initial_size * 100))

        log('Removing methods that exceed max size')
        self.__methods = self.__methods[self.__methods['source'].apply(
            lambda code: len(code) <= constants.MAX_LENGTH)]

        log('{} methods left ({:.2f}%)'.format(
            len(self.__methods), len(self.__methods) / initial_size * 100))


    def __join_tokens(self):
        log('Joining tokens into strings')
        join = lambda tokens: ' '.join(tokens)

        self.train_methods = self.train_methods.applymap(join)
        self.valid_methods = self.valid_methods.applymap(join)
        self.test_methods = self.test_methods.applymap(join)


    def __split_tokens(self):
        log('Splitting strings into tokens')
        split = lambda sentence: str(sentence).split()

        # self.train_methods = self.train_methods.applymap(split)
        # self.valid_methods = self.valid_methods.applymap(split)
        # self.test_methods = self.test_methods.applymap(split)

        self.__methods = self.__methods.applymap(split)


    def __log_split(self):
        log('Train size: {}\n'
            'Validation size: {}\n'
            'Test size: {}'.format(
                len(self.train_methods),
                len(self.valid_methods),
                len(self.test_methods)))


    def __log_vocabs(self):
        log('Number of unique input tokens: {}\n'
            'Number of unique output tokens: {}'.format(
                len(self.input_vocab),
                len(self.output_vocab)))
