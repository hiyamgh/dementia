import pandas as pd
import numpy as np
import os, pickle
from tensorflow.python.platform import flags
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTENC
import category_encoders as ce

FLAGS = flags.FLAGS


def generate_train_test():
    training_path = FLAGS.training_data_path
    testing_path = FLAGS.testing_data_path
    target_variable = FLAGS.target_variable
    cols_drop = FLAGS.cols_drop
    special_encoding = FLAGS.special_encoding
    categorical_cols_path = FLAGS.categorical_columns
    categorical_encoding = FLAGS.categorical_encoding

    # for dropping un-wanted columns
    if cols_drop is not None:
        if special_encoding:
            df_train = pd.read_csv(training_path, encoding=special_encoding).drop(cols_drop, axis=1)
            df_test = pd.read_csv(testing_path, encoding=special_encoding).drop(cols_drop, axis=1)
        else:
            df_train = pd.read_csv(training_path).drop(cols_drop, axis=1)
            df_test = pd.read_csv(testing_path).drop(cols_drop, axis=1)
    else:
        if special_encoding is not None:
            df_train = pd.read_csv(training_path, encoding=special_encoding)
            df_test = pd.read_csv(testing_path, encoding=special_encoding)
        else:
            df_train = pd.read_csv(training_path)
            df_test = pd.read_csv(testing_path)

    if FLAGS.top_features is not None:
        df_fimp = pd.read_csv('input/feature_importance_modified.csv')
        num_feat = int(FLAGS.top_features)
        top_features = list(df_fimp['Feature'])[:num_feat]
        df_train = df_train[top_features + [FLAGS.target_variable]]
        df_test = df_test[top_features + [FLAGS.target_variable]]

    X_train = np.array(df_train.loc[:, df_train.columns != target_variable])
    y_train = np.array(df_train.loc[:, df_train.columns == target_variable])

    if FLAGS.sampling_strategy is not None:
        codebook = pd.read_csv('input/erroneous_codebook_legal_outliers_filtered.csv')
        categorical_indices = get_categorical(df_train.columns, codebook)
        if isfloat(FLAGS.sampling_strategy):
            sm = SMOTENC(random_state=42, categorical_features=categorical_indices, sampling_strategy=float(FLAGS.sampling_strategy))
        else:
            sm = SMOTENC(random_state=42, categorical_features=categorical_indices, sampling_strategy=FLAGS.sampling_strategy)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        X_train, y_train = X_res, y_res
        all_res = np.append(X_res, y_res.reshape(-1, 1), 1)
        if FLAGS.top_features:
            df_train_res = pd.DataFrame(all_res, columns=top_features + ['dem1066'])
        else:
            df_train_res = pd.DataFrame(all_res, columns=list(df_train.columns))
        df_train = df_train_res

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

    if FLAGS.sampling_strategy is None: # because if we re-sampled data, then we transformed training data
        X_train = np.array(df_train.loc[:, df_train.columns != target_variable])
        y_train = np.array(df_train.loc[:, df_train.columns == target_variable])

    X_test = np.array(df_test.loc[:, df_test.columns != target_variable])
    y_test = np.array(df_test.loc[:, df_test.columns == target_variable])

    # scaling the data
    if FLAGS.scaling is not None:
        if FLAGS.scaling == 'min-max':
            scaler = MinMaxScaler()
        elif FLAGS.scaling == 'z-score':
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
