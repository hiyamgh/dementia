""" Code for loading data. """
import pandas as pd
import numpy as np
import os, pickle
import random
from tensorflow.python.platform import flags
from helper import *
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
FLAGS = flags.FLAGS


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        # self.meta_batchsz = FLAGS.meta_batch_size
        self.meta_batchsz = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = FLAGS.num_classes

        # training and testing data
        self.training_path = FLAGS.training_data_path
        self.testing_path = FLAGS.testing_data_path
        self.target_variable = FLAGS.target_variable
        self.cols_drop = FLAGS.cols_drop
        self.special_encoding = FLAGS.special_encoding
        # self.codebook = pd.read_csv('erroneous_codebook_legal_outliers_filtered.csv')
        # self.feature_importances = pd.read_csv('feature_importance.csv')
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

        if FLAGS.sampling_strategy is not None:
            self.codebook = pd.read_csv('erroneous_codebook_legal_outliers_filtered.csv')
            self.feature_importances = pd.read_csv('feature_importance.csv')
            if FLAGS.top_features is not None:
                print('sampling - turned on')
                print('sampling strategy: {}'.format(FLAGS.sampling_strategy))
                top_features = list(self.feature_importances['Feature'])[:FLAGS.top_features]
                self.df_train = self.df_train[top_features+[self.target_variable]]
                self.df_test = self.df_test[top_features+[self.target_variable]]

            categorical_indices = self.get_categorical(self.df_train.columns)
            X_train = np.array(self.df_train.loc[:, self.df_train.columns != self.target_variable])
            y_train = np.array(self.df_train.loc[:, self.df_train.columns == self.target_variable])

            if isfloat(FLAGS.sampling_strategy):
                sm = SMOTENC(random_state=42, categorical_features=categorical_indices, sampling_strategy=float(FLAGS.sampling_strategy))
            else:
                sm = SMOTENC(random_state=42, categorical_features=categorical_indices, sampling_strategy=FLAGS.sampling_strategy)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            self.X_train, self.y_train = X_res, y_res
            all_res = np.append(X_res, y_res.reshape(-1, 1), 1)
            if FLAGS.top_features:
                df_train_res = pd.DataFrame(all_res, columns=top_features+['dem1066'])
            else:
                df_train_res = pd.DataFrame(all_res, columns=list(self.df_train.columns))
            self.df_train = df_train_res

        # create combined dataset for FP growth
        self.df = pd.concat([self.df_train, self.df_test])

        if FLAGS.include_fp == '1':
            print('fp growth - turned on')
            if FLAGS.fp_file is not None and FLAGS.colsmeta_file is not None:
                print('found frequent patterns in {}'.format(FLAGS.fp_file))
                print('found colsmeta          in {}'.format(FLAGS.colsmeta_file))
                with open(FLAGS.fp_file, 'rb') as handle:
                    self.freqItemSet = pickle.load(handle)
                with open(FLAGS.colsmeta_file, 'rb') as handle:
                    self.cols_meta = pickle.load(handle)

                # self.indices_who_has_fp = {}
                # self.indices_without_fp = {}
                # self.indices_who_has_fp['train'] = {}
                # self.indices_who_has_fp['test'] = {}
                # for i, fp in enumerate(self.freqItemSet):
                #     self.indices_who_has_fp['train'][i] = get_fp_indices_raw(fps=fp, df=self.df_train)
                #     self.indices_who_has_fp['test'][i] = get_fp_indices_raw(fps=fp, df=self.df_test)

            else:
                # get the frequent pattern and cols_meta(dictionary containing meta data about columns distribution)
                self.freqItemSet, self.cols_meta = identify_frequent_patterns(df=self.df,
                                                                              target_variable=self.target_variable,
                                                                              supp_fp=FLAGS.supp_fp)
                # if self.freqItemSet:
                #     pass
                # else:
                #     # probably the support was very high, lower it
                #     print('lowering supp_fp from {} to {}'.format(FLAGS.supp_fp, FLAGS.supp_fp - 0.1))
                #     self.freqItemSet, self.cols_meta = identify_frequent_patterns(df=self.df,
                #                                                                   target_variable=self.target_variable,
                #                                                                   supp_fp=FLAGS.supp_fp - 0.1)
                #
                # # dictionary of indices who has fp
                # self.indices_who_has_fp = {}
                # self.indices_without_fp = {}
                # self.indices_who_has_fp['train'] = {}
                # self.indices_who_has_fp['test'] = {}
                # for i, fp in enumerate(self.freqItemSet):
                #     self.indices_who_has_fp['train'][i] = get_fp_indices(fps=fp, cols_meta=self.cols_meta, df=self.df_train)
                #     self.indices_who_has_fp['test'][i] = get_fp_indices(fps=fp, cols_meta=self.cols_meta, df=self.df_test)
                #

            if self.freqItemSet:
                pass
            else:
                # probably the support was very high, lower it
                print('lowering supp_fp from {} to {}'.format(FLAGS.supp_fp, FLAGS.supp_fp - 0.1))
                self.freqItemSet, self.cols_meta = identify_frequent_patterns(df=self.df,
                                                                              target_variable=self.target_variable,
                                                                              supp_fp=FLAGS.supp_fp - 0.1)

                # dictionary of indices who has fp
            self.indices_who_has_fp = {}
            self.indices_without_fp = {}
            self.indices_who_has_fp['train'] = {}
            self.indices_who_has_fp['test'] = {}
            for i, fp in enumerate(self.freqItemSet):
                self.indices_who_has_fp['train'][i] = get_fp_indices(fps=fp, cols_meta=self.cols_meta, df=self.df_train)
                self.indices_who_has_fp['test'][i] = get_fp_indices(fps=fp, cols_meta=self.cols_meta, df=self.df_test)

            if self.freqItemSet:
                    pass
            else:
                # probably the support was very high, lower it
                print('lowering supp_fp from {} to {}'.format(FLAGS.supp_fp, FLAGS.supp_fp - 0.1))
                self.freqItemSet, self.cols_meta = identify_frequent_patterns(df=self.df,
                                                                              target_variable=self.target_variable,
                                                                              supp_fp=FLAGS.supp_fp - 0.1)

                # dictionary of indices who has fp
            self.indices_who_has_fp = {}
            self.indices_without_fp = {}
            self.indices_who_has_fp['train'] = {}
            self.indices_who_has_fp['test'] = {}
            for i, fp in enumerate(self.freqItemSet):
                self.indices_who_has_fp['train'][i] = get_fp_indices(fps=fp, cols_meta=self.cols_meta, df=self.df_train)
                self.indices_who_has_fp['test'][i] = get_fp_indices(fps=fp, cols_meta=self.cols_meta, df=self.df_test)

        else:
            print('fp growth - turned off')

        # training and testing numpy arrays
        if FLAGS.sampling_strategy is None:
            self.X_train = np.array(self.df_train.loc[:, self.df_train.columns != self.target_variable])
            self.y_train = np.array(self.df_train.loc[:, self.df_train.columns == self.target_variable])

        self.X_test = np.array(self.df_test.loc[:, self.df_test.columns != self.target_variable])
        self.y_test = np.array(self.df_test.loc[:, self.df_test.columns == self.target_variable])

        # dictionary containing indices that are not frequent patterns
        # normal to find none, cz frequent patterns are logically in each row
        if FLAGS.include_fp == '1':
            self.indices_without_fp['train'], self.indices_without_fp['test'] = get_non_fp_indices(
                fp2indices_dict=self.indices_who_has_fp,
                x_train=self.X_train,
                x_test=self.X_test)

            if self.indices_without_fp['train'] and self.indices_without_fp['test']:
                # boolean whether there exist indices ho DONT have FPs or not
                self.non_fp_exist = True
            else:
                self.non_fp_exist = False

        self.dim_input = self.X_train.shape[1]
        self.dim_output = self.num_classes

        if FLAGS.scaling is not None:
            if FLAGS.scaling == 'min-max':
                scaler = MinMaxScaler()
            elif FLAGS.scaling == 'z-score':
                scaler = StandardScaler()
            else:
                scaler = RobustScaler()
            # apply the scaling
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

    def get_columns(self, erroneous_codebook):
        # define functions for getting ordinal/categorical
        get_numeric = lambda col_name, row: col_name if row['data_type'] == 'numeric' else -1
        get_ordinal = lambda col_name, row: col_name if row['data_type'] == 'ordinal' else -1
        get_categorical = lambda col_name, row: col_name if row['data_type'] == 'categorical' else -1

        numeric, ordinal, categorical = [], [], []
        for index, row in erroneous_codebook.iterrows():
            col_name = row['COLUMN']
            ordinal.append(get_ordinal(col_name, row))
            categorical.append(get_categorical(col_name, row))
            numeric.append(get_numeric(col_name, row))

        # remove the -1s from lists
        numeric = list(filter((-1).__ne__, numeric))
        ordinal = list(filter((-1).__ne__, ordinal))
        categorical = list(filter((-1).__ne__, categorical))

        return numeric, ordinal, categorical

    def get_categorical(self, columns):
        _, _, categorical = self.get_columns(self.codebook)
        categorical_index = []
        for col in categorical:
            index = np.where(columns == col)
            if len(index) > 0 and len(index[0]) > 0:
                categorical_index.append(int(index[0][0]))
        return categorical_index

    def yield_idxs(self, training=True):
        ''' yield indices, without using fp growth '''
        num_classes = self.num_classes
        if training:
            x, y = self.X_train, self.y_train
        else:
            x, y = self.X_test, self.y_test
        cl_indices = {}
        for cl in range(num_classes):
            # create an entry for the class
            all_idxs_cl = [idx for idx in range(len(y)) if y[idx] == cl]
            cl_indices[cl] = all_idxs_cl

        return cl_indices

    def yield_fp_idxs(self, training=True):
        ''' yield indices, one from each fp '''
        num_classes = self.num_classes
        fp_idxs = {}
        if training:
            y = self.y_train
            indices_who_has_fp = self.indices_who_has_fp['train']
        else:
            y = self.y_test
            indices_who_has_fp = self.indices_who_has_fp['test']

        for cl in range(num_classes):
            # create an entry for the class
            fp_idxs[cl] = {}
            for fp in indices_who_has_fp:
                # get the indices avaialbe for the current frequent pattern
                available_idxs = indices_who_has_fp[fp]
                # get the list of indices for this class that has fp
                all_idxs_cl_fp = [idx for idx in available_idxs if y[idx] == cl]
                fp_idxs[cl][fp] = all_idxs_cl_fp

        # now, we have a dictionary that has the indices of each frequent pattern, in each class
        return fp_idxs

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

    def get_fp_tasks(self, labels, fp_idxs, ifold, nb_samples=None, shuffle=True, training=True):

        # number of frequent patterns
        num_fps = len(fp_idxs[0].keys())

        if nb_samples > num_fps:
            fps_per_sample = nb_samples // num_fps
        elif num_fps > nb_samples:
            fps_per_sample = num_fps // nb_samples
        else:
            fps_per_sample = num_fps

        def get_data_idxs(cl):
            idxs_chosen = []
            num_samples_taken = 0
            while len(idxs_chosen) < nb_samples:
                for fp in fp_idxs[cl]:
                    idxs_chosen.extend(list(np.random.choice(fp_idxs[cl][fp], size=fps_per_sample, replace=True)))
                    num_samples_taken += fps_per_sample

                    if len(idxs_chosen) >= nb_samples:
                        break

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
            idxs_chosen = get_data_idxs(cl=i)
            labels_curr = [i] * len(idxs_chosen)
            labels_curr = np.array([labels_curr, -(np.array(labels_curr)-1)]).T

            all_idxs.extend(idxs_chosen)
            all_labels.extend(labels_curr)

        if shuffle:
            zipped = list(zip(all_idxs, all_labels))
            random.shuffle(zipped)
            all_idxs, all_labels = zip(*zipped)

        return x[all_idxs, :], np.array(all_labels)

    def get_data_tasks(self, labels, data_idxs, ifold, nb_samples=None, shuffle=True, training=True):

        def get_data_idxs(cl):
            if ifold < self.meta_batchsz:
                multiple = 0
                # idxs_chosen = list(np.random.choice(data_idxs[cl][ifold], size=nb_samples, replace=False))
                idxs_chosen = random.sample(data_idxs[cl][ifold], size=nb_samples)
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
            idxs_chosen = get_data_idxs(cl=i)
            labels_curr = [i] * len(idxs_chosen)
            labels_curr = np.array([labels_curr, -(np.array(labels_curr)-1)]).T

            # add the indexes and labels
            all_idxs.extend(idxs_chosen)
            all_labels.extend(labels_curr)

        if shuffle:
            zipped = list(zip(all_idxs, all_labels))
            random.shuffle(zipped)
            all_idxs, all_labels = zip(*zipped)

        # return x[all_idxs, :], np.array(all_labels).reshape(-1, 1)
        return x[all_idxs, :], np.array(all_labels)

    def sample_tasks(self, train):
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
            num_total_batches = 200000
        else:
            num_total_batches = 600

        all_data, all_labels = [], []
        if FLAGS.include_fp == '1':
            fp_idxs = self.yield_fp_idxs(training=train)

        for ifold in range(num_total_batches):

            if FLAGS.include_fp == '1':
                data, labels = self.get_fp_tasks(self.num_classes, fp_idxs, ifold,
                                                 nb_samples=self.num_samples_per_class,
                                                 shuffle=True, training=train)
            else:
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