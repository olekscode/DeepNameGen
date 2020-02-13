import pickle

import torch

from seq2seq import Seq2Seq
import constants

SOURCE_LANG_PATH = constants.LANGS_DIR / 'input_lang.pkl'
TARGET_LANG_PATH = constants.LANGS_DIR / 'output_lang.pkl'
TRAINED_MODEL_PATH = constants.MODELS_DIR / 'trained_model.pt'


class NameGenerator:
    def __init__(self):
        self.__source_lang = self.__load_lang(SOURCE_LANG_PATH)
        self.__target_lang = self.__load_lang(TARGET_LANG_PATH)
        self.__model = self.__build_model()

        trained_model_state = torch.load(str(TRAINED_MODEL_PATH))
        self.__model.load_state_dict(trained_model_state)


    def get_name_and_attention_for(self, method_source):
        input_tensor = self.__sentence_as_tensor(method_source)
        output_tensor, attention = self.__model(input_tensor, return_attention=True)
        return self.__tensor_as_name(output_tensor), attention


    def __load_lang(self, file_path):
        with open(str(file_path), 'rb') as f:
            lang = pickle.load(f)

        return lang


    def __build_model(self):
        return Seq2Seq(
            input_size=self.__source_lang.n_words,
            output_size=self.__target_lang.n_words,
            hidden_size=constants.HIDDEN_SIZE,
            learning_rate=constants.LEARNING_RATE,
            teacher_forcing_ratio=constants.TEACHER_FORCING_RATIO,
            device=constants.DEVICE)


    def __indexes_from_tokens(self, tokens):
        int_tokens = []

        for token in tokens:
            try:
                int_tokens.append(self.__source_lang.word2index[token])
            except KeyError:
                pass

        return int_tokens


    def __tokens_from_indexes(self, indexes):
        return [self.__target_lang.index2word[index] for index in indexes]


    def __tensor_from_indexes(self, indexes):
        return torch.tensor(indexes, dtype=torch.long, device=constants.DEVICE).view(-1, 1)


    def __sentence_as_tensor(self, sentence):
        tokens = sentence.split()
        indexes = self.__indexes_from_tokens(tokens)
        indexes.append(constants.EOS_TOKEN)
        return self.__tensor_from_indexes(indexes)


    def __tensor_as_name(self, tensor):
        tokens = self.__tokens_from_indexes(tensor)
        return ' '.join(tokens[:-1])
