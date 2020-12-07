import pandas as pd
import numpy as np
import os


def stratified_split(df, target_variable, test_ratio, cols_drop=None):
    ''' splits data into train and test in a stratified manner '''
    all_cols = list(df.columns)
    if target_variable not in all_cols:
        raise ValueError('column {} does not exist'.format(target_variable))

    # split into input and output features
    # X, y = df.loc[:, df.columns != target_variable], df.loc[df.columns == target_variable]
    if cols_drop:
        # make sure all passed columns exist in the dataset
        for col in cols_drop:
            if col not in all_cols:
                raise ValueError('Column {} you are trying to drop does not exist'.format(col))
        df = df.drop(cols_drop, axis=1)

    fractions = np.array([1 - test_ratio, test_ratio])

    # shuffle your input
    df = df.sample(frac=1)

    # split into 3 parts
    df_train, df_test = np.array_split(df, (fractions[:-1].cumsum() * len(df)).astype(int))

    return df_train, df_test


def mkdir(output_folder):
    ''' create directory if it does not already exist '''
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


if __name__ == '__main__':
    df = pd.read_csv('input/toy_data/simulated_data.csv')
    df_train, df_test = stratified_split(df,
                                         target_variable='nograd',
                                         test_ratio=0.2,
                                         cols_drop=['id'])

    mkdir(output_folder='input')
    df_train.to_csv(os.path.join('input/toy_data/', 'df_train.csv'), index=False)
    df_test.to_csv(os.path.join('input/toy_data/', 'df_test.csv'), index=False)
