"""
This script is the same as data_preprocessing_without_data_leakage.py but the difference
is that we modified the testing data to contain the 'full validation data (281)'
"""

import pandas as pd
import numpy as np
import pickle
from legalize_data import impute_missing_values
from sklearn.preprocessing import *
from sklearn.impute import KNNImputer
import os
import matplotlib.pyplot as plt
from scipy.special import boxcox


def get_columns(erroneous_codebook):
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


def apply_ordinal_encoding(pooled_ordinal_sub):
    enc = OrdinalEncoder()
    pooled_ordinal_sub = enc.fit_transform(pooled_ordinal_sub)
    return pooled_ordinal_sub


def create_new_category(pooled_categorical_sub):
    ''' create a new category for missing values (takes only categorical columns) '''
    all_cols = list(pooled_categorical_sub.columns)
    df = pooled_categorical_sub

    # loop over all categorical columns
    for col in all_cols:
        vals = list(set(df[col].dropna().values))
        last_category = max(vals)
        new_category = last_category + 1
        df[col] = df[col].fillna(new_category)

    return df


def apply_knn_imputation(pooled_numerical_sub, n_neighbs=5):
    ''' apply knn imputation for missing values (takes only numeric/ordinal columns) '''
    all_cols = list(pooled_numerical_sub.columns)
    # remove the is_validation column from the list so that we do not compute anything for it
    all_cols = list(filter(('is_validation').__ne__, all_cols))
    imputer = KNNImputer(n_neighbors=n_neighbs)
    imputed = imputer.fit_transform(pooled_numerical_sub.drop(['is_validation'], axis=1))
    df = pd.DataFrame(imputed, columns=all_cols)
    # get back the is_validation column
    df['is_validation'] = list(pooled_numerical_sub['is_validation'])
    return df


def replace_erroneous(df, erroneous, numeric, ordinal, categorical):
    ''' Dr. Khalil: 17/3/2021: For the column ANIMALS_2 replace the erroneous by 10
    (the highest threshold) as for the rest, consider them as missing.
    This was for the top 20 column selected by feature importance
    but we will do this for all the columns
    '''
    orig_df = df
    replaced = pd.DataFrame()
    num_ord_cat = numeric + ordinal + categorical
    for col_name in num_ord_cat:
        # get the erroneous values of this column
        err_values = erroneous.loc[erroneous['COLUMN'] == col_name]['erroneous'].values[0]
        if not isinstance(err_values, str):
            if np.isnan(err_values):
                replaced[col_name] = df[col_name]
                continue
            else:
                print('oops, sth is wrong: {}'.format(err_values))
        else:
            col_values = list(df[col_name])
            err_values = list(map(int, err_values.split(',')))
            if col_name == 'ANIMALS_2': # replace the erroneous with 10, the highest threshold
                col_values_replaced = [v if v not in err_values else 10 for v in col_values]
            else:
                col_values_replaced = [v if v not in err_values else np.nan for v in col_values]
            replaced[col_name] = col_values_replaced

    replaced['dem1066'] = df['dem1066']
    replaced['is_validation'] = df['is_validation']
    print('before & after replacing erroneous, are they equal: {}'.format(orig_df.equals(replaced.drop(['is_validation'], axis=1))))
    return replaced


def apply_one_hot_encoding(pooled_categorical_sub):
    all_cols = list(pooled_categorical_sub.columns)
    df = pooled_categorical_sub
    for col in all_cols:
        # apply dummy variable encoding (NOT one hot encoding)
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col, drop_first=True)], axis=1).drop([col], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def apply_dummy_imputation(df):
    return impute_missing_values(df)


def mkdir(directory):
    ''' creates a directory if it does not already exist '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_histograms(df, output_folder):
    mkdir(output_folder)
    all_cols = list(df.columns)
    for col in all_cols:
        plt.hist(df[col], bins=100)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_folder, '{}.png'.format(col)))
        plt.close()


def apply_boxcox(df):
    pos_df = df[df > 0]
    # pp = PowerTransformer()
    bcdata, lam = boxcox(pos_df, 0.5)
    x = np.empty_like(df)
    x[df > 0] = bcdata
    x[df == 0] = -1 / lam


def scale_numeric(df):
    all_cols = list(df.columns)
    scaler = RobustScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=all_cols)


def scale_ordinal(df):
    all_cols = list(df.columns)
    scaler = RobustScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=all_cols)


def scale_data(df_train, df_test, numeric_cols, ordinal_cols, categorical_cols, colnames_train,
               colnames_test, target_var):
    print('scaling training and testing data ...')
    numeric_cols_mod = [c[:-2] if '_2' in c else c for c in numeric_cols]
    ordinal_cols_mod = [c[:-2] if '_2' in c else c for c in ordinal_cols]
    categorical_cols_mod = [c[:-2] if '_2' in c else c for c in categorical_cols]

    numeric_cols_final = list(set(numeric_cols).union(set(numeric_cols_mod)))
    ordinal_cols_final = list(set(ordinal_cols).union(set(ordinal_cols_mod)))
    categorical_cols_final = list(set(categorical_cols).union(set(categorical_cols_mod)))

    imp_feats = list(pd.read_csv('../input/feature_importance_modified.csv')['Feature'])[:20]

    cols2type = {
        'ordinal': [],
        'numeric': [],
        'categorical': []
    }

    df_train = df_train[imp_feats + [target_var]]
    imp_feats_test = [c[:-2] if '_2' in c else c for c in imp_feats]
    df_test = df_test[imp_feats_test + [target_var]]
    df_test = df_test.rename(columns=dict(zip(imp_feats_test, imp_feats)))

    print(df_train.columns)
    print(df_test.columns)

    print(df_train.shape)
    print(df_test.shape)

    for col in df_train.columns:
        if col == target_var:
            continue
        if col in numeric_cols_final:
            cols2type['numeric'].append(col)
        elif col in ordinal_cols_final:
            cols2type['ordinal'].append(col)
        else:
            cols2type['categorical'].append(col)

    df_train1, df_train2, df_test1, df_test2 = None, None, None, None

    if cols2type['numeric'] != [] or cols2type['ordinal'] != []:
        df_train1 = df_train[cols2type['numeric'] + cols2type['ordinal']] # the numeric & ordinal
        df_test1 = df_test[cols2type['numeric'] + cols2type['ordinal']]  # the numeric & ordinal

    if cols2type['categorical'] != []:
        df_train2 = df_train[cols2type['categorical']] # the categorical
        df_test2 = df_test[cols2type['categorical']] # the categorical

    df_train_target = df_train[[target_var]] # the target variable
    df_test_target = df_test[[target_var]] # the target variable

    df_train1_new, df_test1_new = None, None

    if df_train1 is not None and df_test1 is not None:
        scaler = RobustScaler()
        X_train = np.array(df_train1.loc[:, df_train1.columns != target_var]) # the numeric & ordinal
        X_test = np.array(df_test1.loc[:, df_test1.columns != target_var]) # the numeric and ordinal

        X_train = scaler.fit_transform(X_train) # the numeric & ordinal
        X_test = scaler.transform(X_test) # the numeric & ordinal

        df_train1_new = pd.DataFrame(X_train, columns=cols2type['numeric'] + cols2type['ordinal']) # the numeric & ordinal
        df_test1_new = pd.DataFrame(X_test, columns=cols2type['numeric'] + cols2type['ordinal']) # the numeric & ordinal

    to_stack_train, to_stack_test = [], []
    if df_train1_new is not None and df_test1_new is not None:
        to_stack_train.append(df_train1_new)
        to_stack_test.append(df_test1_new)

    if df_train2 is not None and df_test2 is not None:
        to_stack_train.append(df_train2)
        to_stack_test.append(df_test2)

    to_stack_train.append(df_train_target)
    to_stack_test.append(df_test_target)

    df_train_scaled = pd.DataFrame(np.hstack(to_stack_train), columns=cols2type['numeric'] + cols2type['ordinal'] + cols2type['categorical'] + [target_var])
    df_test_scaled = pd.DataFrame(np.hstack(to_stack_test), columns=cols2type['numeric'] + cols2type['ordinal'] + cols2type['categorical'] + [target_var])

    return df_train_scaled, df_test_scaled


def save_cols(numeric, ordinal, categorical, cols_dir):
    # create path if it does not already exist
    mkdir(cols_dir)

    with open(os.path.join(cols_dir, 'numeric.p'), 'wb') as f:
        pickle.dump(numeric, f)
    with open(os.path.join(cols_dir, 'ordinal.p'), 'wb') as f:
        pickle.dump(ordinal, f)
    with open(os.path.join(cols_dir, 'categorical.p'), 'wb') as f:
        pickle.dump(categorical, f)


if __name__ == '__main__':
    # read the erroneous codebook
    erroneous_codebook = pd.read_csv('../input/codebooks/erroneous_codebook_legal_outliers_filtered.csv')
    # pooled = pd.read_csv('../input/pooled_data.csv')
    pooled = pd.read_csv('../input/pooled_data_validation_marked.csv') # read pooled with validation data marked

    with open("../input/codebooks/legal_cols_filtered.txt", "rb") as fp:  # Unpickling
        legal_cols = pickle.load(fp)

        pooled_legal = pooled[legal_cols]
        pooled_legal['is_validation'] = pooled['is_validation']
        pooled_legal['dem1066'] = pooled['dem1066']
        # pooled_legal.to_csv('../input/pooled_legal.csv', index=False)

        numeric, ordinal, categorical = get_columns(erroneous_codebook)
        save_cols(numeric, ordinal, categorical, cols_dir='../input/columns/') # save column names as pickle files

        # replace erroneous values in columns by missing (except for ANIMALS_2)
        pooled_legal_replaced = replace_erroneous(pooled_legal, erroneous_codebook, numeric, ordinal, categorical)

        # data subsets by data type + the target variable + is_validation column
        numeric_df = pooled_legal_replaced[numeric + ['is_validation']]
        ordinal_df = pooled_legal_replaced[ordinal]
        categorical_df = pooled_legal_replaced[categorical]
        target = pooled_legal_replaced[['dem1066']]

        # according to the latest meeting, for the categorical
        # replace missing values by a new category for missing
        categorical_df_imp = create_new_category(pooled_categorical_sub=categorical_df)

        # according to the latest meeting, for the numerical/ordinal
        # impute missing values using KNN
        # gather the ordinal and numeric back into one data frame
        numeric_ordinal = pd.concat([numeric_df, ordinal_df], axis=1)
        numeric_ordinal_imp = apply_knn_imputation(pooled_numerical_sub=numeric_ordinal, n_neighbs=10)

        # gather back imputed datasets into one dataset
        pooled_imputed = pd.concat([numeric_ordinal_imp, categorical_df_imp, target], axis=1)
        # the target variable has empty string '', remove them
        # pooled_imputed = pooled_imputed[pooled_imputed['dem1066'] != '']
        pooled_imputed = pooled_imputed.loc[~pooled_imputed['dem1066'].isin(['', ' '])]
        pooled_imputed.to_csv('../input/pooled_imputed_final.csv', index=False)
        from collections import Counter
        print(Counter(pooled_imputed['dem1066']))

        # split the data into train and test, then apply scaling (to avoid data leakage)
        # df_train, df_test = stratified_split(pooled_imputed, target_variable='dem1066',
        #                                      test_ratio=0.2, cols_drop=None)

        df_train = pooled_imputed[pooled_imputed['is_validation'] == 0]
        df_test = pd.read_csv('../original_datasets/full validation data (281).csv')

        cols_train = list(df_train.columns)
        cols_test = list(df_test.columns)
        # df_imp = pd.read_csv('../input/feature_importance_modified.csv')
        # imp_feats_orig = list(df_imp['Feature'])[:20]
        # imp_feats = [f[:-2] if '_2' in f else f for f in imp_feats_orig]
        # mapping = dict(zip(imp_feats, imp_feats_orig))
        # df_test = df_test.rename(columns=mapping)
        # df_train = df_train[imp_feats_orig + ['dem1066']]

        df_train.drop(['is_validation'], axis=1, inplace=True)
        # df_test.drop(['is_validation'], axis=1, inplace=True)
        #
        # df_train.to_csv('../input/train_imputed_notscaled_validation_marked.csv', index=False)
        # df_test.to_csv('../input/test_imputed_notscaled_validation_marked.csv', index=False)

        print('\ntraining size: {}'.format(len(df_train) / len(pooled_imputed)))
        print('testing size: {}'.format(len(df_test) / len(pooled_imputed)))

        print('\ndf_train')
        print(df_train['dem1066'].value_counts())
        print('0: {}'.format(list(df_train['dem1066']).count('0') / len(df_train)))
        print('1: {}'.format(list(df_train['dem1066']).count('1') / len(df_train)))
        print('df_test')
        print(df_test['dem1066'].value_counts())
        print('0: {}'.format(list(df_test['dem1066']).count('0') / len(df_test)))
        print('1: {}'.format(list(df_test['dem1066']).count('1') / len(df_test)))

        # scale the data (this method scales only the numeric and
        # ordinal columns, in each of the training, then apply to testing
        # in order to avoid data leakage)
        df_train_scaled, df_test_scaled = scale_data(df_train, df_test, numeric_cols=numeric,
                                                     ordinal_cols=ordinal, categorical_cols=categorical,
                                                     colnames_train=cols_train, colnames_test=cols_test,
                                                     target_var='dem1066')

        print('\n Final after scaling')
        print(df_train_scaled.columns)
        print(df_test_scaled.columns)

        print(df_train_scaled.shape)
        print(df_test_scaled.shape)

        df_train_scaled.to_csv('../input/train_imputed_scaled_validation_filtered.csv', index=False)
        df_test_scaled.to_csv('../input/test_imputed_scaled_validation_filtered.csv', index=False)






