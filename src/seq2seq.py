import constants
from encoder import EncoderRNN
from decoder import AttnDecoderRNN
from util import time_str
from logger import log, write_training_log, save_dataframe, plot_and_save_histories

import time
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size,
                 learning_rate, teacher_forcing_ratio, device):
        super(Seq2Seq, self).__init__()

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device

        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size, output_size)

        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)

        self.criterion = nn.NLLLoss()


    def train(self, input_tensor, target_tensor, max_length=constants.MAX_LENGTH):
        encoder_hidden = self.encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length + 1, self.encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[constants.SOS_TOKEN]], device=self.device)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if np.random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di] # Teacher forcing
        else:
            # Without teacher forcing: use its own prediction as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach() # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[di])

                if decoder_input.item() == constants.EOS_TOKEN:
                    break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length


    def trainIters(self, env, evaluator):
        start_total_time = time.time() - env.total_training_time
        start_epoch_time = time.time() # Reset every LOG_EVERY iterations
        start_train_time = time.time() # Reset every LOG_EVERY iterations
        total_loss = 0                 # Reset every LOG_EVERY iterations

        for iter in range(env.iters_completed + 1, constants.NUM_ITER + 1):
            row = env.train_methods.iloc[np.random.randint(len(env.train_methods))]
            input_tensor = row['source']
            target_tensor = row['name']

            loss = self.train(input_tensor, target_tensor)
            total_loss += loss

            if iter % constants.LOG_EVERY == 0:
                log('Completed {} iterations'.format(iter))

                train_time_elapsed = time.time() - start_train_time

                log('Evaluating on validation set')
                start_eval_time = time.time()

                names = evaluator.evaluate(self)

                log('Saving the calculated metrics')
                save_dataframe(names, constants.VALIDATION_NAMES_FILE.format(iter))

                eval_time_elapsed = time.time() - start_eval_time

                env.history = env.history.append({
                    'Loss': total_loss / constants.LOG_EVERY,
                    'Precision': names['Precision'].mean(),
                    'Recall': names['Recall'].mean(),
                    'F1': names['F1'].mean(),
                    'ExactMatch': names['ExactMatch'].mean(),
                    'num_names': len(names['GeneratedName'].unique())
                }, ignore_index=True)

                epoch_time_elapsed = time.time() - start_epoch_time
                total_time_elapsed = time.time() - start_total_time

                env.total_training_time = total_time_elapsed

                history_last_row = env.history.iloc[-1]

                log_dict = OrderedDict([
                    ("Iteration",  '{}/{} ({:.1f}%)'.format(
                        iter, constants.NUM_ITER, iter / constants.NUM_ITER * 100)),
                    ("Average loss", history_last_row['Loss']),
                    ("Average precision", history_last_row['Precision']),
                    ("Average recall", history_last_row['Recall']),
                    ("Average F1", history_last_row['F1']),
                    ("Exact match", history_last_row['ExactMatch']),
                    ("Unique names", int(history_last_row['num_names'])),
                    ("Epoch time", time_str(epoch_time_elapsed)),
                    ("Training time", time_str(train_time_elapsed)),
                    ("Evaluation time", time_str(eval_time_elapsed)),
                    ("Total training time", time_str(total_time_elapsed))
                ])

                write_training_log(log_dict, constants.TRAIN_LOG_FILE)
                plot_and_save_histories(env.history)

                env.iters_completed = iter
                env.save_train()

                # Reseting counters
                total_loss = 0
                start_epoch_time = time.time()
                start_train_time = time.time()


    def forward(self, input_tensor, max_length=constants.MAX_LENGTH, return_attention=False):
        encoder_hidden = self.encoder.initHidden()

        input_length = input_tensor.size(0)

        encoder_outputs = torch.zeros(max_length + 1, self.encoder.hidden_size, device=self.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[constants.SOS_TOKEN]], device=self.device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        attention_vectors = []

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)

            decoded_words.append(topi.item())
            attention_vectors.append(decoder_attention.tolist()[0])

            if decoded_words[-1] == constants.EOS_TOKEN:
                break

            decoder_input = topi.squeeze().detach()

        if return_attention:
            return decoded_words, attention_vectors
        else:
            return decoded_words
