import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from tensorflow.python.platform import flags


training_data_path = '../Advanced-ML-Eval/fake_news_datasets/input/feature_extraction_train_updated.csv'
testing_data_path = '../Advanced-ML-Eval/fake_news_datasets/input/feature_extraction_test_updated.csv'
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

        # training and testing numpy arrays
        self.X_train = np.array(self.df_train.loc[:, self.df_train.columns != self.target_variable])
        self.y_train = np.array(self.df_train.loc[:, self.df_train.columns == self.target_variable])

        self.X_test = np.array(self.df_test.loc[:, self.df_test.columns != self.target_variable])
        self.y_test = np.array(self.df_test.loc[:, self.df_test.columns == self.target_variable])

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
        # self.X_train = scaler.fit_transform(self.X_train)
        # self.X_test = scaler.transform(self.X_test)

        self.dim_input = self.X_train.shape[1]
        self.dim_output = self.num_classes

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

    def make_data_tensor(self, training=True):

        all_data, all_labels = [], []
        for _ in range(self.total_batch_num):
            # sampled_folders, range(self.num_classes), nb_samples=self.nimg
            # 16 in one class, 16 * 2 in one task :)
            data, labels = self.get_data_tasks(self.num_classes, nb_samples=self.nimg, shuffle=True, training=training)
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

