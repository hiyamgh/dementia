from collections import Counter
from numpy import mean
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from data_preprocessing import get_columns
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier, BalancedBaggingClassifier
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import fbeta_score, make_scorer
import pickle, os
import category_encoders as ce
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

f2_score = make_scorer(fbeta_score, beta=2)


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
    else:  # woe
        print('Encoding: {}'.format(enc_method))
        encoder = ce.WOEEncoder(cols=cat_cols)

    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_test_enc = encoder.transform(X_test)
    print(X_train_enc.shape, X_test_enc.shape)
    return X_train_enc, X_test_enc, encoder


def mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# make a prediction with a lof model
def lof_predict(model, trainX, testX):
    composite = np.vstack((trainX, testX))
    yhat = model.fit_predict(composite)
    return yhat[len(trainX):]


def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def brier_skill_score(y, yhat):
    probabilities = [0.01 for _ in range(len(y))]
    brier_ref = metrics.brier_score_loss(y, probabilities)
    bs = metrics.brier_score_loss(y, yhat)
    return 1.0 - (bs / brier_ref)


def split_save(full_path, train_path, test_path):
    df = pd.read_csv(full_path)
    y = np.array(df['dem1066'])
    X = df[df.columns[:-1]]
    split(X, y, train_path, test_path)
    split(X, y, train_path, test_path)


def split(X, y, train_path, test_path):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=36851234)
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X.iloc[list(train_index)], X.iloc[list(test_index)]
        y_train, y_test = y[train_index], y[test_index]

    X_train['dem1066'] = y_train
    X_test['dem1066'] = y_test
    X_train.to_csv(train_path)
    X_test.to_csv(test_path)


def get_categorical(columns):
    codebook = pd.read_csv('../input/codebooks/erroneous_codebook_legal_outliers_filtered.csv')
    _, _, categorical = get_columns(codebook)
    categorical_index = []
    for col in categorical:
        index = np.where(columns == col)
        if len(index) > 0 and len(index[0]) > 0:
            categorical_index.append(int(index[0][0]))
    return categorical_index


def create_pipeline(model, columns, X, enc_method, cat_cols):
    categorical_index = get_categorical(columns)
    print('Encoding: {}'.format(enc_method))

    cat_cols_indxs = [X.columns.get_loc(c) for c in cat_cols if c in X]

    if enc_method == 'catboost':
        transformer = ColumnTransformer(transformers=[('cat', ce.CatBoostEncoder(), cat_cols_indxs)])
    elif enc_method == 'glmm':
        transformer = ColumnTransformer(transformers=[('cat', ce.GLMMEncoder(), cat_cols_indxs)])
    elif enc_method == 'target':
        transformer = ColumnTransformer(transformers=[('cat', ce.TargetEncoder(), cat_cols_indxs)])
    elif enc_method == 'mestimator':
        transformer = ColumnTransformer(transformers=[('cat', ce.MEstimateEncoder(), cat_cols_indxs)])
    elif enc_method == 'james':
        transformer = ColumnTransformer(transformers=[('cat', ce.JamesSteinEncoder(), cat_cols_indxs)])
    else:
        transformer = ColumnTransformer(transformers=[('cat', ce.WOEEncoder(), cat_cols_indxs)])

    steps = [('s', SMOTENC(categorical_index)),
             ('t', transformer),
             ('m', model)]
    pipeline = Pipeline(steps=steps)
    return pipeline


def baseline_logistic(df, test_df):
    X = df[df.columns[:-1]]
    y = df['dem1066']
    test_X = test_df[df.columns[:-1]]
    test_y = test_df['dem1066']
    model = LogisticRegression()
    model.fit(X, y)
    y_predicted = model.predict_proba(test_X)[:, 1]
    print_results('baseline logistic regression', {}, test_y, y_predicted, proba=True, threshold=0.5)


def create_model(y, model, model_name):
    counter = Counter(y)
    reverse_counter = {0: counter[1], 1: counter[0]}
    balance = [{0: 1, 1: 10}, {0: 1, 1: 100}, reverse_counter]
    if model_name in ['Weighted Logistic Regression', 'Weighted Decision Tree Classifier',
                      'Weighted SVM', 'Balanced Random Forest']:
        param_grid = {
            'm__class_weight': balance if model_name != 'Balanced Random Forest' else ['balanced'],
            's__sampling_strategy': ['minority', 'not minority', 'all', 0.5, 1, 0.75],
        }
    elif model_name == 'Weighted XGBoost':
        param_grid = {
            'm__scale_pos_weight': [1, 10, 25, 50, 75, 99, 100, 1000],
            's__sampling_strategy': ['minority', 'not minority', 'all', 0.5, 1, 0.75],
        }
    else:
        param_grid = {
            's__sampling_strategy': ['minority', 'not minority', 'all', 0.5, 1, 0.75],
        }
    return model, param_grid


def print_results(model_name, best_params, y, y_predicted, df_results, encoding_strategy,
                  proba=False, one_class=False, threshold=None, topn=10):
    if proba:
        y_predicted = to_labels(y_predicted, threshold)

    if proba:
        bss = brier_skill_score(y, y_predicted)
        # if one_class:
        #     fscore = fbeta_score(y, y_predicted, beta=2, pos_label=-1)
        # else:
        fscore = fbeta_score(y, y_predicted, beta=2)
        gmean = geometric_mean_score(y, y_predicted, average='weighted')
        pr_auc = metrics.average_precision_score(y, y_predicted)
        if 'm__class_weight' in best_params:
            cost_matrix = best_params['m__class_weight']
        elif 'm__scale_pos_weight' in best_params:
            cost_matrix = best_params['m__scale_pos_weight']
        else:
            cost_matrix = '-'
        sampling_strategy = best_params['s__sampling_strategy']
        df_results = df_results.append({
            'model_name': model_name,
            'f2': '{:.5f}'.format(fscore),
            'gmean': '{:.5f}'.format(gmean),
            'bss': bss,
            'pr_auc': '{:.5f}'.format(pr_auc),
            'sampling_strategy': sampling_strategy,
            'cost_matrix': cost_matrix,
            'encoding_strategy': encoding_strategy
        }, ignore_index=True)
    else:
        if one_class:
            fscore = fbeta_score(y, y_predicted, beta=2, pos_label=-1)
            df_results = df_results.append({
                'model_name': model_name,
                'f2': '{:.5f}'.format(fscore)
            }, ignore_index=True)

        else:
            fscore = fbeta_score(y, y_predicted, beta=2)

            roc = mean(metrics.roc_auc_score(y, y_predicted))
            tn, fp, fn, tp = metrics.confusion_matrix(y, y_predicted).ravel()
            sampling_strategy = best_params['s__sampling_strategy']
            if 'm__class_weight' in best_params:
                cost_matrix = best_params['m__class_weight']
            elif 'm__scale_pos_weight' in best_params:
                cost_matrix = best_params['m__scale_pos_weight']
            else:
                cost_matrix = '-'

            df_results = df_results.append({
                'model_name': model_name,
                'f2': '{:.5f}'.format(fscore),
                'roc_auc': '{:.5f}'.format(roc),
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp,
                'sampling_strategy': sampling_strategy,
                'cost_matrix': cost_matrix,
                'encoding_strategy': encoding_strategy
            }, ignore_index=True)

    return df_results


def test_model_one_class(train_df, test_df, feature_importance, model_class, model_name,
                         df_results, proba=False, one_class=False,
                         topn=10):
    enc_methods = ['catboost', 'glmm', 'target', 'mestimator', 'james', 'woe']

    for enc in enc_methods:
        X = train_df[feature_importance[:topn]]
        y = train_df['dem1066']
        # test_X=test_df[df.columns[:-1]]
        test_X = test_df[feature_importance[:topn]]
        test_y = test_df['dem1066']

        with open('../input/columns/categorical.p', 'rb') as f:
            categorical_cols = pickle.load(f)
        all_cols = list(X.columns)
        cat_cols = [c for c in categorical_cols if c in all_cols]

        X, test_X, _ = encode_categorical_data_supervised(X, y, test_X, cat_cols, enc)
        X['dem1066'] = train_df['dem1066']
        X = X[X['dem1066'] == 0]
        X = X.drop(['dem1066'], axis=1)
        test_y[test_y == 1] = -1
        test_y[test_y == 0] = 1

        estimator = model
        best_params = {}
        estimator.fit(X)

        y_predicted = estimator.predict(test_X)
        df_results = print_results(model_name, best_params, test_y, y_predicted, df_results, enc,
                                   proba=proba, one_class=one_class, topn=topn)

    return df_results


def test_model(train_df, test_df, feature_importance, model_class, model_name,
               df_results, proba=False, one_class=False,
               topn=10):
    if one_class:
        train_df = train_df[train_df['dem1066'] == 0]

    # X=train_df[df.columns[:-1]]
    X = train_df[feature_importance[:topn]]
    y = train_df['dem1066']
    # test_X=test_df[df.columns[:-1]]
    test_X = test_df[feature_importance[:topn]]
    test_y = test_df['dem1066']
    if one_class:
        test_y[test_y == 1] = -1
        test_y[test_y == 0] = 1

    with open('../input/columns/categorical.p', 'rb') as f:
        categorical_cols = pickle.load(f)
    all_cols = list(X.columns)
    cat_cols = [c for c in categorical_cols if c in all_cols]
    enc_methods = ['catboost', 'glmm', 'target', 'mestimator', 'james', 'woe']
    model, param_grid = create_model(y, model_class, model_name)
    for enc in enc_methods:

        if not one_class:
            pipeline = create_pipeline(model, X.columns, X, enc_method=enc, cat_cols=cat_cols)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

            grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring=f2_score)
            grid_result = grid.fit(np.array(X), np.array(y))
            print('Best model: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
            estimator = grid_result.best_estimator_
            best_params = grid_result.best_params_
            estimator.fit(X, y)
        else:
            estimator = model
            best_params = {}
            X, test_X, _ = encode_categorical_data_supervised(X, y, test_X, cat_cols, enc)
            estimator.fit(X)

        # estimator.fit(X, y)
        if proba:
            yhat = estimator.predict_proba(test_X)
            probs = yhat[:, 1]
            # define thresholds
            thresholds = np.arange(0, 1, 0.001)
            # evaluate each threshold
            scores = [fbeta_score(test_y, to_labels(probs, t), beta=2) for t in thresholds]
            # get best threshold
            ix = np.argmax(scores)
            print('Threshold=%.3f, F-measure=%.5f' % (thresholds[ix], scores[ix]))
            best_params['f2_score'] = scores[ix]
            df_results = print_results(model_name, best_params, test_y, probs, df_results, enc,
                                       proba=proba, one_class=one_class,
                                       threshold=thresholds[ix], topn=topn)
        else:
            y_predicted = estimator.predict(test_X)
            # y_predicted=lof_predict(estimator, X, test_X)
            df_results = print_results(model_name, best_params, test_y, y_predicted, df_results, enc,
                                       proba=proba, one_class=one_class, topn=topn)

    return df_results


if __name__ == '__main__':

    df = pd.read_csv('../input/train_imputed_scaled.csv')
    test_df = pd.read_csv('../input/test_imputed_scaled.csv')
    feature_importance = np.array(pd.read_csv('../input/feature_importance_modified.csv')['Feature'])
    # models_dict = {}

    for topn in [10, 20]:

        out_proba = '../output/output_probabilistic/top{}/'.format(topn)
        mkdir(out_proba)

        out_regular = '../output/output_regular/top_{}/'.format(topn)
        mkdir(out_regular)

        out_one_class = '../output/output_one_class/top_{}/'.format(topn)
        mkdir(out_one_class)

        df_proba = pd.DataFrame(columns=['model_name', 'f2', 'gmean', 'bss', 'pr_auc',
                                         'sampling_strategy', 'cost_matrix', 'encoding_strategy'])

        df_regular = pd.DataFrame(columns=['model_name', 'f2', 'tn', 'fp', 'fn', 'tp',
                                           'sampling_strategy', 'cost_matrix', 'encoding_strategy'])

        df_one_class = pd.DataFrame(columns=['model_name', 'f2', 'encoding_strategy'])

        for model, model_name, proba, one_class in [
            (XGBClassifier(), 'Weighted XGBoost', True, False), # scale_pos_weight added by Hiyam
            (LogisticRegression(), 'Weighted Logistic Regression', True, False), # class_weight
            (DecisionTreeClassifier(), 'Weighted Decision Tree Classifier', True, False), # class_weight
            (SVC(), 'Weighted SVM', False, False), # class_weight
            (KNeighborsClassifier(), 'KNeighbors', False, False),
            # (BalancedRandomForestClassifier(), 'Balanced Random Forest', True, False), # class_weight
            (BalancedBaggingClassifier(), 'Balanced Bagging Classifier', True, False), # added by Hiyam
            (EasyEnsembleClassifier(), 'Easy Ensemble Classifier', False, False)]:

            print('Model: {}, proba: {}, one_class: {}'.format(model_name, proba, one_class))

            print('top {}:'.format(topn))
            if proba:
                df_proba = test_model(df, test_df, feature_importance, model, model_name,
                                       df_proba, proba=proba, one_class=one_class, topn=topn)
            else:
                df_regular = test_model(df, test_df, feature_importance, model, model_name,
                                       df_regular, proba=proba, one_class=one_class, topn=topn)

            print('===================================================================\n')

            if proba:
                df_proba = df_proba.sort_values(by='f2', ascending=False)
                df_proba.to_csv(os.path.join(out_proba, 'prob_results.csv'), index=False)
            else:
                df_regular = df_regular.sort_values(by='f2', ascending=False)
                df_regular.to_csv(os.path.join(out_regular, 'regular_results.csv'), index=False)

        # now for one class classification
        for model, model_name, proba, one_class in [
            (OneClassSVM(gamma='scale', nu=0.1), 'One Class SVM', False, True),
            (EllipticEnvelope(contamination=0.1), 'EllipticEnvelope', False, True),
            (IsolationForest(contamination=0.1), 'IsolationForest', False, True),
            (LocalOutlierFactor(contamination=0.1, novelty=True), 'LocalOutlierFactor', False, True)]:
            print('======================= Model: {} ======================='.format(model_name))
            df_one_class = test_model_one_class(df, test_df, feature_importance, model, model_name,
                                                df_one_class, proba=proba, one_class=one_class, topn=topn)

            df_one_class = df_one_class.sort_values(by='f2', ascending=False)
            df_one_class.to_csv(os.path.join(out_one_class, 'one_class_results.csv'), index=False)


