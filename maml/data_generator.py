import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import *
from helper import *
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from tensorflow.python.platform import flags


training_data_path = 'input/feature_extraction_train_updated.csv'
testing_data_path = 'input/feature_extraction_test_updated.csv'
df_train = pd.read_csv(training_data_path, encoding='latin-1')
df_test = pd.read_csv(testing_data_path, encoding='latin-1')
cols_drop = ['article_title', 'article_content', 'source', 'source_category', 'unit_id']
target_variable = 'label'
scaling = 'robust'


# the data frames
df = pd.concat([df_train, df_test]).drop(cols_drop, axis=1).sample(frac=1).reset_index(drop=True)
df_train = df_train.drop(cols_drop, axis=1)
df_test = df_test.drop(cols_drop, axis=1)
special_encoding = 'latin-1'


class DataGenerator(object):

    def __init__(self, nway, kshot, kquery, meta_batchsz, total_batch_num = 200000):
        self.meta_batchsz = meta_batchsz
        # number of images to sample per class
        self.nimg = kshot + kquery
        self.num_classes = nway
        self.total_batch_num = total_batch_num

        # training and testing data
        self.training_path = training_data_path
        self.testing_path = testing_data_path
        self.target_variable = target_variable
        self.cols_drop = cols_drop
        self.special_encoding = special_encoding

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

        # X, Y = make_classification(n_samples=50000, n_features=10, n_informative=8,
        #                            n_redundant=0, n_clusters_per_class=2)
        #
        # X_train, X_test, y_train, y_test = train_test_split(X, Y)

        # create combined dataset for FP growth
        self.df = pd.concat([self.df_train, self.df_test])
        # get the frequent pattern and cols_meta(dictionary containing meta data about columns distribution)
        self.freqItemSet, self.cols_meta = identify_frequent_patterns(df=df, target_variable=target_variable)

        # dictionary of indices for each frequent pattern
        self.indices_who_has_fp = {}
        self.indices_without_fp = {}
        self.indices_who_has_fp['train'] = {}
        self.indices_who_has_fp['test'] = {}
        for i, fp in enumerate(self.freqItemSet):
            self.indices_who_has_fp['train'][i] = get_fp_indices(fps=fp, cols_meta=self.cols_meta, df=self.df_train)
            self.indices_who_has_fp['test'][i] = get_fp_indices(fps=fp, cols_meta=self.cols_meta, df=self.df_test)

        # training and testing numpy arrays
        self.X_train = np.array(self.df_train.loc[:, self.df_train.columns != self.target_variable])
        self.y_train = np.array(self.df_train.loc[:, self.df_train.columns == self.target_variable])

        self.X_test = np.array(self.df_test.loc[:, self.df_test.columns != self.target_variable])
        self.y_test = np.array(self.df_test.loc[:, self.df_test.columns == self.target_variable])

        # dictionary containing indices that are not frequent patterns
        # normal to find none, cz frequent patterns are logically in each row
        self.indices_without_fp['train'], self.indices_without_fp['test'] = get_non_fp_indices(
            fp2indices_dict=self.indices_who_has_fp,
            x_train=self.X_train,
            x_test=self.X_test)

        if self.indices_without_fp['train'] and self.indices_without_fp['test']:
            # boolean whether there exist indices ho DONT have FPs or not
            self.non_fp_exist = True
        else:
            self.non_fp_exist = False

        # self.X_train = X_train
        # self.y_train = y_train
        # self.X_test = X_test
        # self.y_test = y_test

        # if scaling == 'robust':
        #     scaler = RobustScaler()
        # elif scaling == 'minmax':
        #     scaler = MinMaxScaler()
        # else:
        #     scaler = StandardScaler()
        #

        scaler = MinMaxScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        self.dim_input = self.X_train.shape[1]
        self.dim_output = self.num_classes

    def yield_fp_idxs(self, training=True):
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

    #             for i in range(self.meta_batchsz):
    #                 if i != self.meta_batchsz - 1:
    #                     data_idxs[cl][i] = list(range(i*num_data_per_batch, (i+1)*num_data_per_batch))
    #                 else:
    #                     data_idxs[cl][i+1] = list(range(i*num_data_per_batch, data_len))

    # 0, 1, 2, 3,
    # 4-4, 5-4, 6-4, 7-4,
    # 8-4*2, 9-4*2, 10-4*2, 11-4*2

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
                    idxs_chosen.extend(list(np.random.choice(fp_idxs[cl][fp], size=fps_per_sample, replace=False)))
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

        # return x[all_idxs, :], np.array(all_labels).reshape(-1, 1)
        return x[all_idxs, :], np.array(all_labels)

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

    def make_data_tensor(self, training=True):

        all_data, all_labels = [], []
        fp_idxs = self.yield_fp_idxs(training=training)
        # data_idxs = self.yield_data_idxs(training=training)

        for ifold in range(self.total_batch_num):
            # sampled_folders, range(self.num_classes), nb_samples=self.nimg
            # 16 in one class, 16 * 2 in one task :)

            # data, labels = self.get_data_tasks(self.num_classes, data_idxs, ifold, nb_samples=self.nimg, shuffle=True, training=training)
            data, labels = self.get_fp_tasks(self.num_classes, fp_idxs, ifold, nb_samples=self.nimg, shuffle=True, training=training)

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


if __name__ == '__main__':
    training_data_path = 'input/feature_extraction_train_updated.csv'
    testing_data_path = 'input/feature_extraction_test_updated.csv'
    df_train = pd.read_csv(training_data_path, encoding='latin-1')
    df_test = pd.read_csv(testing_data_path, encoding='latin-1')
    cols_drop = ['article_title', 'article_content', 'source', 'source_category', 'unit_id']
    target_variable = 'label'
    scaling = 'robust'

    # the data frames
    df = pd.concat([df_train, df_test]).drop(cols_drop, axis=1).sample(frac=1).reset_index(drop=True)
    df_train = df_train.drop(cols_drop, axis=1)
    df_test = df_test.drop(cols_drop, axis=1)
    special_encoding = 'latin-1'

    freqItemSet, cols_meta = identify_frequent_patterns(df=df, target_variable=target_variable)
    indices_who_has_fp = get_fp_indices(fps=freqItemSet[0], cols_meta=cols_meta, df=df)
    print(indices_who_has_fp)
