import pandas as pd
import numpy as np
import os, pickle
import random
from tensorflow.python.platform import flags
from helper import *
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from categorical_encoding import encode_categorical_data_supervised, encode_categorical_data_unsupervised
FLAGS = flags.FLAGS


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


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
        self.cat_encoding = FLAGS.categorical_encoding

        # for dropping un-wanted columns
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
            self.codebook = pd.read_csv('input/erroneous_codebook_legal_outliers_filtered.csv')
            self.feature_importances = pd.read_csv('input/feature_importance_modified.csv')
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

            else:
                # get the frequent pattern and cols_meta(dictionary containing meta data about columns distribution)
                self.freqItemSet, self.cols_meta = identify_frequent_patterns(df=self.df,
                                                                              target_variable=self.target_variable,
                                                                              supp_fp=FLAGS.supp_fp)

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

        # encode categorical data (after FP Growth)
        df_orig = self.df
        if self.cat_encoding not in ['binary', 'basen', 'sum', 'backward_diff', 'polynomial', 'count', 'helmert',
                                     'catboost', 'glmm', 'target', 'mestimator', 'james', 'woe']:
            raise ValueError('categorical encoding \'{}\' is not supported'.format(self.cat_encoding))
        else:
            with open('../input/columns/categorical.p', 'rb') as f:
                categorical_cols = pickle.load(f)
            categorical_cols = [c for c in categorical_cols if c in df_orig.columns]
            if self.cat_encoding in ['binary', 'basen', 'sum', 'backward_diff', 'polynomial', 'count', 'helmert']:
                df_enc, encoder = encode_categorical_data_unsupervised(df_orig, categorical_cols, self.cat_encoding)
                self.df = df_enc
                # since df is just pd.concat([df_train, df_test]), then df_train is first n rows
                # and df_test is the last n rows
                df_train = self.df.head(len(self.df_train))
                df_test = self.df.tail(len(self.df) - len(df_train))
                # update training and testing data after encoding
                self.df_train = df_train
                self.df_test = df_test
            else:
                Xtraindf = self.df_train.drop([self.target_variable], axis=1)
                y_train_list = list(self.df_train[self.target_variable])
                ytraindf = self.df_train[[self.target_variable]]
                Xtestdf = self.df_test.drop([self.target_variable], axis=1)
                y_test_list = list(self.df_test[self.target_variable])

                Xtraindf_enc, Xtestdf_enc, encoder = encode_categorical_data_supervised(Xtraindf, ytraindf,Xtestdf, categorical_cols, self.cat_encoding)

                df_train = Xtraindf_enc
                df_train[self.target_variable] = y_train_list
                df_test = Xtestdf_enc
                df_test[self.target_variable] = y_test_list

                self.df_train = df_train
                self.df_test = df_test

            # save the encoder, so that later we can do inverse_transform()
            encoding_folder = 'encoders/'
            mkdir(encoding_folder)
            with open(os.path.join(encoding_folder, '{}_encoder.p'.format(self.cat_encoding)), 'wb') as f:
                pickle.dump(encoder, f)

        # training and testing numpy arrays
        # if FLAGS.sampling_strategy is None:
        #     self.X_train = np.array(self.df_train.loc[:, self.df_train.columns != self.target_variable])
        #     self.y_train = np.array(self.df_train.loc[:, self.df_train.columns == self.target_variable])

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
                # get the indices available for the current frequent pattern
                available_idxs = indices_who_has_fp[fp]
                # get the list of indices for this class that has fp
                all_idxs_cl_fp = [idx for idx in available_idxs if y[idx] == cl]
                fp_idxs[cl][fp] = all_idxs_cl_fp

        # now, we have a dictionary that has the indices of each frequent pattern, in each class
        return fp_idxs

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

        x = self.X_train if training else self.X_test

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