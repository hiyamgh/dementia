import category_encoders as ce
import pandas as pd
import pickle


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


def encode_categorical_data_unsupervised(df, cat_cols, enc_method):
    if enc_method == 'binary':
        encoder = ce.BinaryEncoder(cols=cat_cols)
    elif enc_method == 'basen':
        encoder = ce.BaseNEncoder(cols=cat_cols, base=4)
    elif enc_method == 'sum':
        encoder = ce.SumEncoder(cols=cat_cols)
    elif enc_method == 'backward_diff':
        encoder = ce.BackwardDifferenceEncoder(cols=cat_cols)
    elif enc_method == 'polynomial':
        encoder = ce.PolynomialEncoder(cols=cat_cols)
    elif enc_method == 'count':
        encoder = ce.CountEncoder(cols=cat_cols)
    else:
        encoder = ce.HelmertEncoder(cols=cat_cols)

    df_enc = encoder.fit_transform(df)
    return df_enc, encoder


if __name__ == '__main__':
    df_train = pd.read_csv('../input/train_imputed_scaled.csv')
    df_test = pd.read_csv('../input/test_imputed_scaled.csv')
    fimp = pd.read_csv('../input/feature_importance_modified.csv')
    erroneous_codebook = pd.read_csv('../input/codebooks/erroneous_codebook_legal_outliers_filtered.csv')
    pooled = pd.read_csv('../input/pooled_data.csv')
    target_variable = 'dem1066'

    top20 = list(fimp['Feature'])[:10]
    df = pd.concat([df_train, df_test])
    df = df[top20]
    df_train = df_train[top20 + ['dem1066']]
    df_test = df_test[top20 + ['dem1066']]

    with open('../input/columns/categorical.p', 'rb') as f:
        categorical = pickle.load(f)
    cat_top20 = [c for c in categorical if c in top20]

    Xtraindf = df_train.drop([target_variable], axis=1)
    ytraindf = df_train[[target_variable]]
    Xtestdf = df_test.drop([target_variable], axis=1)

    for enc in ['catboost', 'glmm', 'target', 'mestimator', 'james', 'woe']:
        encode_categorical_data_supervised(Xtraindf, ytraindf, Xtestdf, cat_top20, enc_method=enc)