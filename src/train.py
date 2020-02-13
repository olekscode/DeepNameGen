import constants
from environment import Environment
from preprocessing import TensorEncoder
from evaluator import Evaluator
from logger import log, save_dataframe

import time


def load_last_saved_environment():
    try:
        env = Environment()
        env.load()
        return env
    except Exception as e:
        log('Error: ' + str(e))
        log('Failed to load the enviroment')
        return initialize_new_enviroment()


def initialize_new_enviroment():
    env = Environment()
    env.initialize_new()
    env.save_data()
    return env


def start_training():
    try:
        if constants.START_FROM_SCRATCH:
            env = initialize_new_enviroment()
        else:
            env = load_last_saved_environment()

        log('Converting text to tensors')

        encoder = TensorEncoder(env.input_vocab, env.output_vocab)

        env.train_methods = encoder.encode(env.train_methods, 'source', ['name'])
        env.valid_methods = encoder.encode(env.valid_methods, 'source', ['name'])
        env.test_methods = encoder.encode(env.test_methods, 'source', ['name'])

        log('Initializing evaluators')
        valid_evaluator = Evaluator(env.valid_methods, encoder)
        test_evaluator = Evaluator(env.test_methods, encoder)

        log('Training the model')
        env.model.trainIters(env, valid_evaluator)

        log('Saving the environment')
        env.save_train()

        log('Evaluating on test set')
        names = test_evaluator.evaluate(env.model)
        save_dataframe(names, constants.TEST_NAMES_FILE)

        log('Done')

    except Exception as e:
        # Log the error message and raise it again so see more info
        log("Error: " + str(e))
        raise e
