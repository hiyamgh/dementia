""" Code for loading data. """
import random, os, pickle
import numpy as np
from tensorflow.python.platform import flags
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import category_encoders as ce

FLAGS = flags.FLAGS


class DataGenerator(object):
    def __init__(self, num_samples_per_class, batch_size, config={}):
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

        # if we are taking top 10/20/etc features selected by feature selection
        if FLAGS.top_features is not None:
            df_fimp = pd.read_csv('input/feature_importance_modified.csv')
            top_features = list(df_fimp['Feature'])[:FLAGS.top_features]
            self.df_train = self.df_train[top_features + [FLAGS.target_variable]]
            self.df_test = self.df_test[top_features + [FLAGS.target_variable]]

        target_variable = FLAGS.target_variable
        categorical_cols_path = FLAGS.categorical_columns
        categorical_encoding = FLAGS.categorical_encoding

        self.X_train = np.array(self.df_train.loc[:, self.df_train.columns != target_variable])
        self.y_train = np.array(self.df_train.loc[:, self.df_train.columns == target_variable])

        if FLAGS.sampling_strategy is not None:
            codebook = pd.read_csv('input/erroneous_codebook_legal_outliers_filtered.csv')
            categorical_indices = self.get_categorical(self.df_train.columns, codebook)
            if self.isfloat(FLAGS.sampling_strategy):
                sm = SMOTENC(random_state=42, categorical_features=categorical_indices,
                             sampling_strategy=float(FLAGS.sampling_strategy))
            else:
                sm = SMOTENC(random_state=42, categorical_features=categorical_indices,
                             sampling_strategy=FLAGS.sampling_strategy)
            X_res, y_res = sm.fit_resample(self.X_train, self.y_train)
            self.X_train, self.y_train = X_res, y_res
            all_res = np.append(X_res, y_res.reshape(-1, 1), 1)
            if FLAGS.top_features:
                df_train_res = pd.DataFrame(all_res, columns=top_features + ['dem1066'])
            else:
                df_train_res = pd.DataFrame(all_res, columns=list(self.df_train.columns))
            self.df_train = df_train_res

        # if we want to encode categorical data
        if categorical_encoding is not None:
            if categorical_encoding not in ['catboost', 'glmm', 'target', 'mestimator', 'james', 'woe']:
                raise ValueError('categorical encoding \'{}\' is not supported'.format(categorical_encoding))
            with open(categorical_cols_path, 'rb') as f:
                categorical_cols = pickle.load(f)
            all_cols = list(self.df_test.columns)
            df_train, df_test, encoder = self.encode_datasets(self.df_train, self.df_test, target_variable,
                                                         cat_cols=[c for c in categorical_cols if c in all_cols],
                                                         cat_enc=categorical_encoding)

            # save the encoder, so that later we can do inverse_transform()
            encoding_folder = 'encoders/'
            self.mkdir(encoding_folder)
            with open(os.path.join(encoding_folder, '{}_encoder.p'.format(categorical_encoding)), 'wb') as f:
                pickle.dump(encoder, f)

        if FLAGS.sampling_strategy is None:  # because if we re-sampled data, then we transformed training data
            self.X_train = np.array(self.df_train.loc[:, self.df_train.columns != target_variable])
            self.y_train = np.array(self.df_train.loc[:, self.df_train.columns == target_variable])

        self.X_test = np.array(self.df_test.loc[:, self.df_test.columns != target_variable])
        self.y_test = np.array(self.df_test.loc[:, self.df_test.columns == target_variable])

        if FLAGS.scaling is not None:
            if FLAGS.scaling == 'min-max':
                scaler = MinMaxScaler()
            elif FLAGS.scaling == 'z-score':
                scaler = StandardScaler()
            else:
                scaler = RobustScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

    def isfloat(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def mkdir(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def get_categorical(self, columns, codebook):
        _, _, categorical = self.get_columns(codebook)
        categorical_index = []
        for col in categorical:
            index = np.where(columns == col)
            if len(index) > 0 and len(index[0]) > 0:
                categorical_index.append(int(index[0][0]))
        return categorical_index

    def encode_datasets(self, df_train, df_test, target_variable, cat_cols, cat_enc):
        # transform the train /test inputs and outputs into data frames which is needed for category_encoders module
        Xtraindf = df_train.drop([target_variable], axis=1)
        ytraindf = df_train[target_variable]
        Xtestdf = df_test.drop([target_variable], axis=1)
        ytestdf = df_test[target_variable]

        # get the encoded training and testing datasets
        Xtraindf_enc, Xtestdf_enc, encoder = self.encode_categorical_data_supervised(Xtraindf, ytraindf, Xtestdf,
                                                                                cat_cols, cat_enc)
        df_train = Xtraindf_enc
        df_train[target_variable] = list(ytraindf)
        df_test = Xtestdf_enc
        df_test[target_variable] = list(ytestdf)

        return df_train, df_test, encoder

    def encode_categorical_data_supervised(self, X_train, y_train, X_test, cat_cols, enc_method):

        if enc_method == 'catboost':
            print('Encoding: {}'.format(enc_method))
            encoder = ce.CatBoostEncoder(cols=cat_cols)
        elif enc_method == 'glmm':
            print('Encoding: {}'.format(enc_method))
            encoder = ce.GLMMEncoder(cols=cat_cols)
        elif enc_method == 'target':
            print('Encoding: {}'.format(enc_method))
            encoder = ce.TargetEncoder(cols=cat_cols)
        elif enc_method == 'mestimator':
            print('Encoding: {}'.format(enc_method))
            encoder = ce.MEstimateEncoder(cols=cat_cols)
        elif enc_method == 'james':
            print('Encoding: {}'.format(enc_method))
            encoder = ce.JamesSteinEncoder(cols=cat_cols)
        else:  # woe
            print('Encoding: {}'.format(enc_method))
            encoder = ce.WOEEncoder(cols=cat_cols)

        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_test_enc = encoder.transform(X_test)
        print(X_train_enc.shape, X_test_enc.shape)
        return X_train_enc, X_test_enc, encoder

    def get_columns(self, erroneous_codebook):
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