""" Code for loading data. """
import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """

        # batch size 32, num_classes 1, num_samples_per_class 2
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = config.get('num_classes', FLAGS.num_classes)  # num classes now 5 (5-way)?

        # training and testing data
        self.training_path = FLAGS.training_path
        self.testing_path = FLAGS.testing_path
        self.target_variable = FLAGS.target_variable
        self.cols_drop = FLAGS.cols_drop
        self.special_encoding = FLAGS.special_encoding

        if self.cols_drop is not None:
            if self.special_encoding:
                self.df_train = pd.read_csv(self.training_path, encoding=self.special_encoding).drop(self.cols_drop, axis=1)
                self.df_test = pd.read_csv(self.testing_path, encoding=self.special_encoding).drop(self.cols_drop, axis=1)
            else:
                self.df_train = pd.read_csv(self.training_path).drop(self.cols_drop, axis=1).drop(self.cols_drop, axis=1)
                self.df_test = pd.read_csv(self.testing_path).drop(self.cols_drop, axis=1).drop(self.cols_drop, axis=1)
        else:
            if self.special_encoding:
                self.df_train = pd.read_csv(self.training_path, encoding=self.special_encoding)
                self.df_test = pd.read_csv(self.testing_path, encoding=self.special_encoding)
            else:
                self.df_train = pd.read_csv(self.training_path).drop(self.cols_drop, axis=1)
                self.df_test = pd.read_csv(self.testing_path).drop(self.cols_drop, axis=1)
        
        # training and testing numpy arrays
        self.X_train = np.array(self.df_train.loc[:, self.df_train.columns != self.target_variable])
        self.y_train = np.array(self.df_train.loc[:, self.df_train.columns == self.target_variable])

        self.X_test = np.array(self.df_test.loc[:, self.df_test.columns != self.target_variable])
        self.y_test = np.array(self.df_test.loc[:, self.df_test.columns == self.target_variable])

    def generate_episode_data(self, x, y, n_shots):
        data = []
        labels = []
        for cls in range(self.num_classes):
            idxs_curr = [idx for idx in range(len(x)) if y[idx] == cls]
            idxs_chosen = np.random.choice(range(len(idxs_curr)), size=n_shots, replace=False)
            data.append(x[idxs_chosen])
            labels.append([cls] * len(idxs_chosen))

        all_data = ([data[i] for i in range(len(data))])

        return np.concatenate(all_data), np.array(labels).reshape(-1, 1)

    def make_data_tensor(self, train=True):
        if train:
            num_total_batches = 100
        else:
            num_total_batches = 60

        # make list of files
        print('Generating filenames')
        all_data = []
        all_data_labels = []
        # in each batch (episode we must say), for each class create num_samples_per_class
        for _ in range(self.batch_size):
            if train:
                data, labels = self.generate_episode_data(x=self.X_train, y=self.y_train, n_shots=self.num_samples_per_class)
            else:
                data, labels = self.generate_episode_data(x=self.X_test, y=self.y_test, n_shots=self.num_samples_per_class)

            all_data.append(data)
            all_data_labels.append(labels)

        return all_data, all_data_labels