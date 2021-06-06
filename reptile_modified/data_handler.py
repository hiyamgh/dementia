import pandas as pd
import numpy as np
import os
import random
from tensorflow.python.platform import flags
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
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


def generate_train_test():
    training_path = FLAGS.training_data_path
    testing_path = FLAGS.testing_data_path
    target_variable = FLAGS.target_variable
    cols_drop = FLAGS.cols_drop
    special_encoding = FLAGS.special_encoding

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

    # create combined dataset for FP growth
    # df = pd.concat([df_train, df_test])

    X_train = np.array(df_train.loc[:, df_train.columns != target_variable])
    y_train = np.array(df_train.loc[:, df_train.columns == target_variable])

    X_test = np.array(df_test.loc[:, df_test.columns != target_variable])
    y_test = np.array(df_test.loc[:, df_test.columns == target_variable])

    # dim_input = X_train.shape[1]
    # dim_output = num_classes

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