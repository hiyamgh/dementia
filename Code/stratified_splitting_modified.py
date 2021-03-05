import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import os


def stratified_split(df, target_variable, test_ratio, cols_drop=None):
    print('splitting data into training and testing -- stratified')
    col_names = list(df.columns)
    col_names.remove(target_variable)
    if cols_drop is not None:
        df = df.drop(cols_drop, axis=1)

    # X = np.array(df.loc[:, df.columns != target_variable])
    # y = np.array(df.loc[:, df.columns == target_variable])

    X = df.loc[:, df.columns != target_variable]
    y = df.loc[:, df.columns == target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=test_ratio)

    df_train = X_train
    df_train[target_variable] = y_train[target_variable]

    df_test = X_test
    df_test[target_variable] = y_test[target_variable]

    counts_train = Counter(df_train[target_variable])
    counts_test = Counter(df_test[target_variable])

    print('df_train percentages:')
    for k, v in counts_train.items():
        print('{}: {}%'.format(k, (v/len(df_train)) * 100))
    print('\ndf_test percentages:')
    for k, v in counts_test.items():
        print('{}: {}%'.format(k, (v/len(df_test)) * 100))

    return df_train, df_test


def mkdir(output_folder):
    ''' create directory if it does not already exist '''
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


# if __name__ == '__main__':
#     df = pd.read_csv('input/toy_data/simulated_data.csv')
#     df_train, df_test = stratified_split(df, target_variable='nograd', test_ratio=0.2, cols_drop=['id'])
#     mkdir(output_folder='input')
#     df_train.to_csv(os.path.join('input/toy_data/', 'df_train.csv'), index=False)
#     df_test.to_csv(os.path.join('input/toy_data/', 'df_test.csv'), index=False)
