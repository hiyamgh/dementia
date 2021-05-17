"""
Loading and augmenting the Omniglot dataset.

To use these APIs, you should prepare a directory that
contains all of the alphabets from both images_background
and images_evaluation.
"""

import os, pickle
import random
from PIL import Image
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import category_encoders as ce


def load_datasets(args):
    """
    Applies pre-processing to training and testing datasets.
    * Encodes categorical columns, if specified in args parser
    * Drops un-wanted columns, if specified in args parser
    * Takes top 'n' features selected by feature selection, if specified in args parser
    * Over/Under samples training dataset, if specified in args parser
    * Scales data according to good standards avoiding data leakage, if specified in args parser
    :param args: the parsed arguments
    :return:
    """
    # if we want to drop any columns
    if args.cols_drop is not None:
        if args.special_encoding:
            df_train = pd.read_csv(args.training_data_path, encoding=args.special_encoding).drop(args.cols_drop, axis=1)
            df_test = pd.read_csv(args.testing_data_path, encoding=args.special_encoding).drop(args.cols_drop, axis=1)
        else:
            df_train = pd.read_csv(args.training_data_path).drop(args.cols_drop, axis=1)
            df_test = pd.read_csv(args.testing_data_path).drop(args.cols_drop, axis=1)
    else:
        if args.special_encoding is not None:
            df_train = pd.read_csv(args.training_data_path, encoding=args.special_encoding)
            df_test = pd.read_csv(args.testing_data_path, encoding=args.special_encoding)
        else:
            df_train = pd.read_csv(args.training_data_path)
            df_test = pd.read_csv(args.testing_data_path)

    # if we are taking top 10/20/etc features selected by feature selection
    if args.top_features is not None:
        df_fimp = pd.read_csv('input/feature_importance_modified.csv')
        num_feat = int(args.top_features)
        top_features = list(df_fimp['Feature'])[:num_feat]
        df_train = df_train[top_features + [args.target_variable]]
        df_test = df_test[top_features + [args.target_variable]]

    target_variable = args.target_variable
    categorical_cols_path = args.categorical_columns
    categorical_encoding = args.categorical_encoding

    X_train = np.array(df_train.loc[:, df_train.columns != target_variable])
    y_train = np.array(df_train.loc[:, df_train.columns == target_variable])

    if args.sampling_strategy is not None:
        codebook = pd.read_csv('input/erroneous_codebook_legal_outliers_filtered.csv')
        categorical_indices = get_categorical(df_train.columns, codebook)
        if isfloat(args.sampling_strategy):
            sm = SMOTENC(random_state=42, categorical_features=categorical_indices, sampling_strategy=float(args.sampling_strategy))
        else:
            sm = SMOTENC(random_state=42, categorical_features=categorical_indices, sampling_strategy=args.sampling_strategy)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        X_train, y_train = X_res, y_res
        all_res = np.append(X_res, y_res.reshape(-1, 1), 1)
        if args.top_features:
            df_train_res = pd.DataFrame(all_res, columns=top_features + ['dem1066'])
        else:
            df_train_res = pd.DataFrame(all_res, columns=list(df_train.columns))
        df_train = df_train_res

    # if we want to encode categorical data
    if categorical_encoding is not None:
        if categorical_encoding not in ['catboost', 'glmm', 'target', 'mestimator', 'james', 'woe']:
            raise ValueError('categorical encoding \'{}\' is not supported'.format(categorical_encoding))
        with open(categorical_cols_path, 'rb') as f:
            categorical_cols = pickle.load(f)
        all_cols = list(df_test.columns)
        df_train, df_test, encoder = encode_datasets(df_train, df_test, target_variable,
                                                 cat_cols=[c for c in categorical_cols if c in all_cols],
                                                 cat_enc=categorical_encoding)

        # save the encoder, so that later we can do inverse_transform()
        encoding_folder = 'encoders/'
        mkdir(encoding_folder)
        with open(os.path.join(encoding_folder, '{}_encoder.p'.format(categorical_encoding)), 'wb') as f:
            pickle.dump(encoder, f)

    if args.sampling_strategy is None: # because if we re-sampled data, then we transformed training data
        X_train = np.array(df_train.loc[:, df_train.columns != target_variable])
        y_train = np.array(df_train.loc[:, df_train.columns == target_variable])

    X_test = np.array(df_test.loc[:, df_test.columns != target_variable])
    y_test = np.array(df_test.loc[:, df_test.columns == target_variable])

    if args.scaling is not None:
        if args.scaling == 'min-max':
            scaler = MinMaxScaler()
        elif args.scaling == 'z-score':
            scaler = StandardScaler()
        else:
            scaler = RobustScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_categorical(columns, codebook):
    _, _, categorical = get_columns(codebook)
    categorical_index = []
    for col in categorical:
        index = np.where(columns == col)
        if len(index) > 0 and len(index[0]) > 0:
            categorical_index.append(int(index[0][0]))
    return categorical_index


def get_columns(erroneous_codebook):
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


def encode_datasets(df_train, df_test, target_variable, cat_cols, cat_enc):
    # transform the train /test inputs and outputs into data frames which is needed for category_encoders module
    Xtraindf = df_train.drop([target_variable], axis=1)
    ytraindf = df_train[target_variable]
    Xtestdf = df_test.drop([target_variable], axis=1)
    ytestdf = df_test[target_variable]

    # get the encoded training and testing datasets
    Xtraindf_enc, Xtestdf_enc, encoder = encode_categorical_data_supervised(Xtraindf, ytraindf, Xtestdf,
                                                                            cat_cols, cat_enc)
    df_train = Xtraindf_enc
    df_train[target_variable] = list(ytraindf)
    df_test = Xtestdf_enc
    df_test[target_variable] = list(ytestdf)

    return df_train, df_test, encoder


def encode_categorical_data_supervised(X_train, y_train, X_test, cat_cols, enc_method):

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
    else: # woe
        print('Encoding: {}'.format(enc_method))
        encoder = ce.WOEEncoder(cols=cat_cols)

    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_test_enc = encoder.transform(X_test)
    print(X_train_enc.shape, X_test_enc.shape)
    return X_train_enc, X_test_enc, encoder


def read_dataset(data_dir):
    """
    Iterate over the characters in a data directory.

    Args:
      data_dir: a directory of alphabet directories.

    Returns:
      An iterable over Characters.

    The dataset is unaugmented and not split up into
    training and test sets.
    """
    for alphabet_name in sorted(os.listdir(data_dir)):
        alphabet_dir = os.path.join(data_dir, alphabet_name)
        if not os.path.isdir(alphabet_dir):
            continue
        for char_name in sorted(os.listdir(alphabet_dir)):
            if not char_name.startswith('character'):
                continue
            yield Character(os.path.join(alphabet_dir, char_name), 0)

# def split_dataset(dataset, num_train=1200):
def split_dataset(dataset, num_train=50):
    """
    Split the dataset into a training and test set.

    Args:
      dataset: an iterable of Characters.

    Returns:
      A tuple (train, test) of Character sequences.
    """
    all_data = list(dataset)
    random.shuffle(all_data)
    return all_data[:num_train], all_data[num_train:]

def augment_dataset(dataset):
    """
    Augment the dataset by adding 90 degree rotations.

    Args:
      dataset: an iterable of Characters.

    Returns:
      An iterable of augmented Characters.
    """
    for character in dataset:
        for rotation in [0, 90, 180, 270]:
            yield Character(character.dir_path, rotation=rotation)

# pylint: disable=R0903
class Character:
    """
    A single character class.
    """
    def __init__(self, dir_path, rotation=0):
        self.dir_path = dir_path
        self.rotation = rotation
        self._cache = {}

    def sample(self, num_images):
        """
        Sample images (as numpy arrays) from the class.

        Returns:
          A sequence of 28x28 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.png')]
        random.shuffle(names)
        images = []
        for name in names[:num_images]:
            images.append(self._read_image(os.path.join(self.dir_path, name)))
        return images

    def _read_image(self, path):
        if path in self._cache:
            return self._cache[path]
        with open(path, 'rb') as in_file:
            img = Image.open(in_file).resize((28, 28)).rotate(self.rotation)
            self._cache[path] = np.array(img).astype('float32')
            return self._cache[path]
