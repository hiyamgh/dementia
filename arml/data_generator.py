""" Code for loading data. """
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import pickle
from utils import get_images
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import ipdb

FLAGS = flags.FLAGS


class DataGenerator(object):
    def __init__(self, num_samples_per_class, batch_size, config={}):
        # self.batch_size = batch_size
        # self.num_samples_per_class = num_samples_per_class
        # self.num_classes = 1  # by default 1 (only relevant for classification problems)

        self.meta_batchsz = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = FLAGS.num_classes

        # training and testing data
        self.training_path = FLAGS.training_data_path
        self.testing_path = FLAGS.testing_data_path
        self.target_variable = FLAGS.target_variable
        self.cols_drop = FLAGS.cols_drop
        self.special_encoding = FLAGS.special_encoding

        if self.cols_drop is not None:
            if self.special_encoding:
                self.df_train = pd.read_csv(self.training_path, encoding=self.special_encoding).drop(self.cols_drop, axis=1)
                self.df_test = pd.read_csv(self.testing_path, encoding=self.special_encoding).drop(self.cols_drop, axis=1)
            else:
                self.df_train = pd.read_csv(self.training_path).drop(self.cols_drop, axis=1)
                self.df_test = pd.read_csv(self.testing_path).drop(self.cols_drop, axis=1)
        else:
            if self.special_encoding is not None:
                self.df_train = pd.read_csv(self.training_path, encoding=self.special_encoding)
                self.df_test = pd.read_csv(self.testing_path, encoding=self.special_encoding)
            else:
                self.df_train = pd.read_csv(self.training_path)
                self.df_test = pd.read_csv(self.testing_path)

        self.X_train = np.array(self.df_train.loc[:, self.df_train.columns != self.target_variable])
        self.y_train = np.array(self.df_train.loc[:, self.df_train.columns == self.target_variable])

        self.X_test = np.array(self.df_test.loc[:, self.df_test.columns != self.target_variable])
        self.y_test = np.array(self.df_test.loc[:, self.df_test.columns == self.target_variable])

        self.dim_input = self.X_train.shape[1]
        self.dim_output = self.num_classes

        if FLAGS.scaling is not None:
            if FLAGS.scaling == 'min-max':
                scaler = MinMaxScaler()
            elif FLAGS.scaling == 'z-score':
                scaler = StandardScaler()
            else:
                scaler = RobustScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

    def sample_tasks(self, train):
        all_idxs, all_labels = [], []
        if train:
            x, y = self.X_train, self.y_train
        else:
            x, y = self.X_test, self.y_test

        for i in range(self.num_classes):
            idxs = [idx for idx in range(len(x)) if y[idx] == i]
            idxs_chosen = random.sample(idxs, self.num_samples_per_class)
            labels_curr = [i] * len(idxs_chosen)
            labels_curr = np.array([labels_curr, -(np.array(labels_curr) - 1)]).T

            all_idxs.extend(idxs_chosen)
            all_labels.extend(labels_curr)

        zipped = list(zip(all_idxs, all_labels))
        random.shuffle(zipped)
        all_idxs, all_labels = zip(*zipped)

        return x[all_idxs, :], np.array(all_labels)

    def make_data_tensor(self, train=True):

        num_total_batches = 200000 if train else 600

        all_data, all_labels = [], []
        for ifold in range(num_total_batches):

            data, labels = self.sample_tasks(train=train)

            all_data.extend(data)
            all_labels.extend(labels)

        examples_per_batch = self.num_classes * self.num_samples_per_class  # 2*16 = 32

        all_data_batches, all_label_batches = [], []
        for i in range(self.meta_batchsz):  # 4 .. i.e. 4 * examples_per-batch = 4 * 32 = 128
            # current task, 128 data points
            data_batch = all_data[i * examples_per_batch:(i + 1) * examples_per_batch]
            labels_batch = all_labels[i * examples_per_batch: (i + 1) * examples_per_batch]

            all_data_batches.append(np.array(data_batch))
            all_label_batches.append(np.array(labels_batch))

        all_image_batches = np.array(all_data_batches) # (4, 32, nb_cols)
        all_label_batches = np.array(all_label_batches) # (4, 32, 1)

        return all_image_batches, all_label_batches