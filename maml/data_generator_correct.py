""" Code for loading data. """
import os
import random
import tensorflow as tf
import pandas as pd

from tensorflow.python.platform import flags
from utils import get_images
from helper import *

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
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = FLAGS.num_classes
        self.meta_batchsz = FLAGS.meta_batch_size
        # self.num_classes = 1  # by default 1 (only relevant for classification problems)

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

        # create combined dataset for FP growth
        self.df = pd.concat([self.df_train, self.df_test])

        # training and testing numpy arrays
        self.X_train = np.array(self.df_train.loc[:, self.df_train.columns != self.target_variable])
        self.y_train = np.array(self.df_train.loc[:, self.df_train.columns == self.target_variable])

        self.X_test = np.array(self.df_test.loc[:, self.df_test.columns != self.target_variable])
        self.y_test = np.array(self.df_test.loc[:, self.df_test.columns == self.target_variable])

        self.dim_input = self.X_train.shape[1]
        self.dim_output = self.num_classes

    def get_sample_data(self, train):
        all_idxs = []
        all_labels = []
        if train:
            x = self.X_train
            y = self.y_train
        else:
            x = self.X_test
            y = self.y_test

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
        if train:
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            num_total_batches = 200
        else:
            num_total_batches = 600

        # make list of files
        print('Generating filenames')
        all_data, all_labels = [], []
        for _ in range(num_total_batches):
            # sampled_character_folders = random.sample(folders, self.num_classes)
            # random.shuffle(sampled_character_folders)
            # labels_and_images = get_images(sampled_character_folders, range(self.num_classes), nb_samples=self.num_samples_per_class, shuffle=False)
            data, labels = self.get_sample_data(train=train)

            all_data.extend(data)
            all_labels.extend(labels)

        all_data_batches, all_label_batches = [], []
        examples_per_batch = self.num_classes * self.num_samples_per_class
        for i in range(self.meta_batchsz):  # 4 .. i.e. 4 * examples_per-batch = 4 * 32 = 128
            # current task, 128 data points
            data_batch = all_data[i * examples_per_batch:(i + 1) * examples_per_batch]
            labels_batch = all_labels[i * examples_per_batch: (i + 1) * examples_per_batch]

            all_data_batches.append(np.array(data_batch))
            all_label_batches.append(np.array(labels_batch))

        all_image_batches = np.array(all_data_batches)  # (4, 32, nb_cols)
        all_label_batches = np.array(all_label_batches)  # (4, 32, 1)

        return all_image_batches, all_label_batches

    # def generate_sinusoid_batch(self, train=True, input_idx=None):
    #     # Note train arg is not used (but it is used for omniglot method.
    #     # input_idx is used during qualitative testing --the number of examples used for the grad update
    #     amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
    #     phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
    #     outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
    #     init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
    #     for func in range(self.batch_size):
    #         init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
    #         if input_idx is not None:
    #             init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)
    #         outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
    #     return init_inputs, outputs, amp, phase
