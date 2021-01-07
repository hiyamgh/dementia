import numpy as np
import pandas as pd
import os, sys
import random
import tensorflow as tf
import tqdm
import pickle
from tensorflow.python.platform import flags
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

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

    def __init__(self, num_samples_per_class, batch_size):
        self.meta_batchsz = batch_size
        # number of images to sample per class
        self.nimg = num_samples_per_class
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

        # specify dimensions of input and output for neural network (feed forward)
        self.dim_input = self.X_train.shape[1]
        self.dim_output = self.num_classes

        # # scale the data
        # if FLAGS.scaling == 'robust':
        #     scaler = RobustScaler()
        # elif FLAGS.scaling == 'minmax':
        #     scaler = MinMaxScaler()
        # else:
        #     scaler = StandardScaler()
        #
        # self.X_train = scaler.fit_transform(self.X_train)
        # self.X_test = scaler.transform(self.X_test)

    def yield_data_idxs(self, training=True):
        num_classes = self.num_classes
        data_idxs = {}
        if training:
            x = self.X_train
            y = self.y_train
        else:
            x = self.X_test
            y = self.y_test
        for cl in range(num_classes):
            data_idxs[cl] = {}
            data_len = len(x)
            idxs_for_class = [idx for idx in range(len(x)) if y[idx] == cl]
            num_data_per_batch = len(idxs_for_class) // self.meta_batchsz

            for i in range(self.meta_batchsz):
                if i != self.meta_batchsz - 1:
                    curr_list = idxs_for_class[i*num_data_per_batch: (i+1)*num_data_per_batch]
                    data_idxs[cl][i] = curr_list
                else:
                    curr_list = idxs_for_class[i*num_data_per_batch: data_len]
                    data_idxs[cl][i] = curr_list

        return data_idxs

    #             for i in range(self.meta_batchsz):
    #                 if i != self.meta_batchsz - 1:
    #                     data_idxs[cl][i] = list(range(i*num_data_per_batch, (i+1)*num_data_per_batch))
    #                 else:
    #                     data_idxs[cl][i+1] = list(range(i*num_data_per_batch, data_len))

    # 0, 1, 2, 3,
    # 4-4, 5-4, 6-4, 7-4,
    # 8-4*2, 9-4*2, 10-4*2, 11-4*2

    def get_data_tasks(self, labels, data_idxs, ifold, nb_samples=None, shuffle=True, training=True):

        def get_data_idxs(cl):
            if ifold < self.meta_batchsz:
                multiple = 0
                idxs_chosen = list(np.random.choice(data_idxs[cl][ifold], size=nb_samples, replace=False))
            else:
                if ifold%self.meta_batchsz == 0:
                    multiple = ifold/self.meta_batchsz
                else:
                    possibles = list(range(ifold))
                    for num in reversed(possibles):
                        if num%self.meta_batchsz == 0:
                            multiple = num/self.meta_batchsz
                            break
                idxs_chosen = list(np.random.choice(data_idxs[cl][ifold - self.meta_batchsz*multiple], size=nb_samples, replace=False))

            return idxs_chosen

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
            # idxs_curr = [idx for idx in range(len(x)) if y[idx] == i]
            # sample nb_samples data points per class i
            # idxs_chosen = sampler(idxs_curr)
            idxs_chosen = get_data_idxs(cl=i)
            labels_curr = [i] * len(idxs_chosen)
            labels_curr = np.array([labels_curr, -(np.array(labels_curr)-1)]).T
            # add the indexes and labels
            all_idxs.extend(idxs_chosen)
            # all_labels.extend([i] * len(idxs_chosen))
            all_labels.extend(labels_curr)

        # flatten the list of indxs
        # all_idxs = [item for sublist in all_idxs for item in sublist]
        # all_labels = [item for sublist in all_idxs for item in sublist]
        if shuffle:
            zipped = list(zip(all_idxs, all_labels))
            random.shuffle(zipped)
            all_idxs, all_labels = zip(*zipped)

        # return x[all_idxs, :], np.array(all_labels).reshape(-1, 1)
        return x[all_idxs, :], np.array(all_labels)

    def make_data_tensor(self, train=True):

        all_data, all_labels = [], []
        data_idxs = self.yield_data_idxs(training=train)
        self.total_batch_num = 1000
        # if training:
        #     self.total_batch_num = 200000
        # else:
        #     self.total_batch_num = 1000

        for ifold in range(self.total_batch_num):
            # sampled_folders, range(self.num_classes), nb_samples=self.nimg
            # 16 in one class, 16 * 2 in one task :)
            data, labels = self.get_data_tasks(self.num_classes, data_idxs, ifold, nb_samples=self.nimg, shuffle=True, training=train)
            all_data.extend(data)
            all_labels.extend(labels)

        examples_per_batch = self.num_classes * self.nimg  # 2*16 = 32

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



