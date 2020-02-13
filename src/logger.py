import constants

# TODO: Move elsewhere
from metrics import confusion_dataframe
from exceptions import DriveSecretsNotFound

from visualizations import (
    plot_loss, plot_precision, plot_recall, plot_f1, plot_exact_match,
    plot_all_scores, plot_num_names)

import os
import datetime
import pickle
from io import StringIO

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.settings import InvalidConfigError

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    'log', 'save_dataframe', 'save_image', 'save_pickle',
    'use_default_logger', 'use_drive_logger', 'is_drive_logger',
    'write_training_log', 'plot_and_save_histories'
]

__logger = None


def use_default_logger():
    global __logger
    __logger = DefaultLogger()


def use_drive_logger():
    global __logger
    __logger = DriveLogger(constants.DRIVE_DIR)


# TODO: Remove this. Move the logic that asks for the type of loger to this file
def is_drive_logger():
    return type(__logger) == DriveLogger


def log(string, fname=constants.LOG_FILE, append=True):
    __logger.log(string, fname, append)


def save_dataframe(df, fname):
    __logger.save_dataframe(df, fname)


def read_dataframe(fname):
    return __logger.read_dataframe(fname)


def save_image(fig, fname, update=True):
    __logger.save_image(fig, fname, update)


def save_pickle(obj, fname):
    __logger.save_pickle(obj, fname)


def write_training_log(log_dict, fname):
    log_template = ": {}\n".join(log_dict.keys()) + ": {}"
    log_string = log_template.format(*log_dict.values())

    log(log_string, fname=fname, append=True)


def plot_and_save_histories(histories):
    fig = plot_loss(histories['Loss'])
    save_image(fig, constants.LOSS_IMG_FILE)

    fig = plot_precision(histories['Precision'])
    save_image(fig, constants.PRECISION_IMG_FILE)

    fig = plot_recall(histories['Recall'])
    save_image(fig, constants.RECALL_IMG_FILE)

    fig = plot_f1(histories['F1'])
    save_image(fig, constants.F1_IMG_FILE)

    fig = plot_exact_match(histories['ExactMatch'])
    save_image(fig, constants.EXACT_MATCH_IMG_FILE)

    fig = plot_all_scores(
        histories['Precision'],
        histories['Recall'],
        histories['F1'],
        histories['ExactMatch'])
    save_image(fig, constants.ALL_SCORES_IMG_FILE)

    fig = plot_num_names(histories['num_names'])
    save_image(fig, constants.NUM_NAMES_IMG_FILE)


class DefaultLogger:
    def __init__(self):
        self.log_files = []


    def write_log(self, string, fname, append):
        self.remove_old_file_on_first_log(fname)

        with open(str(fname), 'a') as f:
            f.write(string)

        print(string)


    def log(self, string, fname, append):
        string = '[{}]\n{}\n\n'.format(datetime.datetime.now(), string)
        self.write_log(string, fname, append)


    def save_dataframe(self, df, fname):
        df.to_csv(fname, sep='\t', index=False)
        self.on_file_saved(str(fname))


    def read_dataframe(self, fname):
        return pd.read_csv(fname, sep='\t')


    def save_image(self, fig, fname, update):
        fig.savefig(fname)
        plt.close(fig)
        self.on_file_saved(str(fname))


    def save_pickle(self, obj, fname):
        with open(str(fname), 'wb') as f:
            pickle.dump(obj, f)

        self.on_file_saved(str(fname))


    def on_file_saved(self, fname):
        """DefaultLogger foes nothing, DriveLogger uploads file to Drive"""
        pass


    def remove_old_file_on_first_log(self, fname):
        # Is this a first log?
        if fname not in self.log_files:
            self.log_files.append(fname)

            try:
                # Remove if exists
                os.remove(str(fname))
            except OSError:
                # Otherwise do nothing
                pass


class DriveLogger(DefaultLogger):
    """Inspired by: https://gist.github.com/macieksk/038b201a54d9e804d1b5"""

    def __init__(self, folder_name):
        super(DriveLogger, self).__init__()

        self.__login()
        self.__folder = self.__find_folders(folder_name)[0]

        self.__files = {}
        self.__logs = {}


    def write_log(self, string, fname, append):
        """Writes a log string to the file on Google Drive.

        If file with this name doesn't exist it gets created.

        Parameters
        ----------

        string : str
            A string to be written

        fname : str
            File name on Google Drive

        append : bool
            Append to the end of existing file or rewrite it

        Examples
        --------

        >>> from drive import Drive

        Login and authenticate.

        >>> drive = Drive('NameGen')

        Create a new file and write several logs to it.

        >>> drive.log('Lorem ipsum', 'lipsum.txt')
        >>> drive.log('dolor sit', 'lipsum.txt')
        >>> drive.log('amet', 'lipsum.txt')

        Create a new file and override it with each log

        >>> drive.log('Hello', 'hello.txt')
        >>> drive.log('world', 'hello.txt', append=False)
        """

        super(DriveLogger, self).write_log(string, fname, append)
        fname = os.path.basename(str(fname))

        if fname not in self.__files.keys():
            self.__files[fname] = self.__create_file(fname)
            self.__logs[fname] = []

        if not append:
            # Clear the log
            self.__logs[fname] = []

        self.__logs[fname].append(string)
        self.__files[fname].SetContentString(''.join(self.__logs[fname]))
        self.__files[fname].Upload()


    def on_file_saved(self, fname):
        self.__upload_file(fname, update=True)


    def __login(self):
        try:
            self.__gauth = GoogleAuth()
            self.__gauth.LocalWebserverAuth()        # Creates local webserver and auto handles authentication
            self.__drive = GoogleDrive(self.__gauth) # Create GoogleDrive instance with authenticated GoogleAuth instance
        except InvalidConfigError:
            raise DriveSecretsNotFound


    def __find_folders(self, fldname):
        file_list = self.__drive.ListFile({
            'q': "title='{}' and mimeType contains 'application/vnd.google-apps.folder' and trashed=false".format(fldname)
            }).GetList()
        return file_list


    def __create_file(self, name):
        param = {
            'title': name,
            'parents': [{ u'id': self.__folder['id'] }]
        }

        return self.__drive.CreateFile(param)


    def __upload_file(self, fname, update=True):
        if fname not in self.__files.keys() or not update:
            self.__files[fname] = self.__create_file(
                os.path.basename(str(fname)))

        self.__files[fname].SetContentFile(fname)
        self.__files[fname].Upload()


if constants.WRITE_LOGS_TO_GOOGLE_DRIVE:
    use_drive_logger()
else:
    use_default_logger()
