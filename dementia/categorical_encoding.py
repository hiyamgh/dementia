'''
@author: Hiyam K. Ghannam
@email: hkg02@mail.aub.edu
'''

import category_encoders as ce
import pandas as pd
import pickle


def encode_categorical_data(df, cat_cols, enc_method):
    if any(cat_cols) not in df.columns:
        raise ValueError('some columns you passed are not present in the dataset')
    if enc_method == 'binary':
        encoder = ce.BinaryEncoder(cols=cat_cols)
        df_enc = encoder.fit_transform(df)
        return df_enc
    elif enc_method == 'basen':
        encoder = ce.BaseNEncoder(cols=cat_cols, base=4)
        df_enc = encoder.fit_transform(df)
        return df_enc
    elif enc_method == 'sum':
        encoder = ce.SumEncoder(cols=cat_cols)
        df_enc = encoder.fit_transform(df)
        return df_enc
    elif enc_method == 'backward_diff':
        encoder = ce.BackwardDifferenceEncoder(cols=cat_cols)
        df_enc = encoder.fit_transform(df)
        return df_enc
    elif enc_method == 'polynomial':
        encoder = ce.PolynomialEncoder(cols=cat_cols)
        df_enc = encoder.fit_transform(df)
        return df_enc
    elif enc_method == 'count':
        encoder = ce.CountEncoder(cols=cat_cols)
        df_enc = encoder.fit_transform(df)
        return df_enc
    # elif enc_method == 'hashing':
    else:
        encoder = ce.HelmertEncoder(cols=cat_cols)
        df_enc = encoder.fit_transform(df)
        return df_enc


if __name__ == '__main__':
    df_train = pd.read_csv('../input/train_imputed_scaled.csv')
    df_test = pd.read_csv('../input/test_imputed_scaled.csv')
    fimp = pd.read_csv('../input/feature_importance_modified.csv')
    erroneous_codebook = pd.read_csv('../input/codebooks/erroneous_codebook_legal_outliers_filtered.csv')
    pooled = pd.read_csv('../input/pooled_data.csv')

    top20 = list(fimp['Feature'])[:20]
    df = pd.concat([df_train, df_test])
    df = df[top20]

    with open('../input/columns/categorical.p', 'rb') as f:
        categorical = pickle.load(f)
    cat_top20 = [c for c in categorical if c in top20]


