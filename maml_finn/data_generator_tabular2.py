import numpy as np
import pandas as pd
import os, sys
import random
import tensorflow as tf
import tqdm
import pickle
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

def get_images(paths, labels, nb_samples=None, shuffle=True):
	if nb_samples is not None:
		sampler = lambda x: random.sample(x, nb_samples)
	else:
		sampler = lambda x: x
	images = [(i, os.path.join(path, image)) \
	          for i, path in zip(labels, paths) \
	          for image in sampler(os.listdir(path))]
	if shuffle:
		random.shuffle(images)
	return images




class DataGenerator(object):

    def __init__(self, batch_size):
        self.meta_batchsz = batch_size
        # number of images to sample per class
        self.nimg = FLAGS.kshot + FLAGS.kquery
        self.num_classes = FLAGS.num_classes

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

    def get_data_tasks(self, labels, nb_samples=None, shuffle=True, training=True):

        if nb_samples is not None:
            sampler = lambda idxs: np.random.choice(range(len(idxs)), size=nb_samples, replace=False)
        else:
            sampler = lambda idxs: idxs

        if training:
            x = self.X_train
            y = self.y_train
        else:
            x = self.X_test
            y = self.y_test

        all_idxs = []
        all_labels = []
        for i in range(labels):
            # get all the data that belong to a particular class
            idxs_curr = [idx for idx in range(len(x)) if y[idx] == i]
            # sample nb_samples data points per class i
            idxs_chosen = sampler(idxs_curr)
            # add the indexes and labels
            all_idxs.extend(idxs_chosen)
            all_labels.extend([i] * len(idxs_chosen))

        # flatten the list of indxs
        # all_idxs = [item for sublist in all_idxs for item in sublist]
        # all_labels = [item for sublist in all_idxs for item in sublist]
        if shuffle:
            zipped = list(zip(all_idxs, all_labels))
            random.shuffle(zipped)
            all_idxs, all_labels = zip(*zipped)

        return x[all_idxs, :], np.array(all_labels).reshape(-1, 1)

    def make_data_tensor(self, train=True):
        if train:
            num_total_batches = 100
        else:
            num_total_batches = 60

        all_data, all_labels = [], []
        for _ in range(num_total_batches):
            # sampled_folders, range(self.num_classes), nb_samples=self.nimg
            # 16 in one class, 16 * 2 in one task :)
            data, labels = self.get_data_tasks(self.num_classes, nb_samples=self.nimg, shuffle=True, training=train)
            all_data.extend(data)
            all_labels.extend(labels)

        examples_per_batch = self.num_classes * self.nimg  # 2*16 = 32

        all_data_batches, all_label_batches = [], []
        for i in range(self.meta_batchsz):  # 4
            # current task, 128 data points
            data_batch = all_data[i * examples_per_batch:(i + 1) * examples_per_batch]
            labels_batch = all_labels[i * examples_per_batch: (i + 1) * examples_per_batch]

            all_data_batches.append(np.array(data_batch))
            all_label_batches.append(np.array(labels_batch))

        all_image_batches = np.array(all_data_batches) # (4, 32, nb_cols)
        all_label_batches = np.array(all_label_batches) # (4, 32, 1)

        return all_image_batches, all_label_batches

