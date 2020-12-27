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
        last_category = max(df[col])
        new_category = last_category + 1
        df[col] = df[col].fillna(new_category)

    return df


def apply_knn_imputation(pooled_numerical_sub, n_neighbs=5):
    ''' apply knn imputation for missing values (takes only numeric/ordinal columns) '''
    all_cols = list(pooled_numerical_sub.columns)
    imputer = KNNImputer(n_neighbors=n_neighbs)
    imputed = imputer.fit_transform(pooled_numerical_sub)
    return pd.DataFrame(imputed, columns=all_cols)


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


if __name__ == '__main__':
    # read the erroneous codebook
    erroneous_codebook = pd.read_csv('../input/codebooks/erroneous_codebook_legal_outliers_filtered.csv')
    pooled = pd.read_csv('../input/pooled_data.csv')

    with open("../input/codebooks/legal_cols_filtered.txt", "rb") as fp:  # Unpickling
        legal_cols = pickle.load(fp)

        pooled_legal = pooled[legal_cols]
        pooled_legal['dem1066'] = pooled['dem1066']
        # pooled_legal.to_csv('../input/pooled_legal.csv', index=False)

        numeric, ordinal, categorical = get_columns(erroneous_codebook)

        # data subsets by data type + the target variable
        numeric_df = pooled_legal[numeric]
        ordinal_df = pooled_legal[ordinal]
        categorical_df = pooled_legal[categorical]
        target = pooled_legal[['dem1066']]

        # according to the latest meeting, for the categorical
        # replace missing values by a new category for missing
        categorical_df_imp = create_new_category(pooled_categorical_sub=categorical_df)

        # according to the latest meeting, for the numerical/ordinal
        # impute missing values using KNN
        # gather the ordinal and numeric back into one data frame
        numeric_ordinal = pd.concat([numeric_df, ordinal_df], axis=1)
        numeric_ordinal_imp = apply_knn_imputation(pooled_numerical_sub=numeric_ordinal, n_neighbs=10)

        # scaling the data
        # scale numeric and ordinal with Robust scaler
        numeric_ordinal_imp = scale_numeric(df=numeric_ordinal_imp)

        # concatenate all subsets back together
        pooled_imputed = pd.concat([numeric_ordinal_imp, categorical_df_imp, target], axis=1)

        # save the data
        pooled_imputed.to_csv('../input/pooled_imputed_scaled.csv', index=False)

        # remove missing values using dummy imputation (impute by replacing
        # with the majority) + apply scaling
        # numeric_df_imp = apply_dummy_imputation(df=numeric_df)

        # # generate histograms of all numeric columns
        # # apply_boxcox(numeric_df_imp)
        # generate_histograms(numeric_df_imp, output_folder='../output/numeric/before_boxcox/')
        #
        # # scale numeric data
        # numeric_df_imp = scale_numeric(df=numeric_df_imp)
        #
        # # remove missing values using dummy imputation (impute by replacing
        # # with the majority) + apply scaling
        # ordinal_df_imp = apply_dummy_imputation(df=ordinal_df)
        # ordinal_df_imp = scale_ordinal(df=ordinal_df_imp)
        # # ordinal encoding
        # ordinal_df_imp = apply_ordinal_encoding(pooled_ordinal_sub=ordinal_df_imp)
        # ordinal_df_imp = pd.DataFrame(ordinal_df_imp, columns=ordinal)
        #
        # # # remove missing values using dummy imputation
        # categorical_df_imp = apply_dummy_imputation(df=categorical_df)
        # # one hot encoding
        # categorical_df_imp = apply_one_hot_encoding(pooled_categorical_sub=categorical_df_imp)
        #
        # # save processed data
        # pooled_processed = pd.concat([numeric_df_imp, ordinal_df_imp, categorical_df_imp, target], axis=1)
        # pooled_processed.to_csv('../input/pooled_proc_scimp.csv', index=False)







