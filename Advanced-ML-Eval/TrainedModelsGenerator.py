import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
import os
import pickle


class ModelGenerator:
    ''' class that trains models and saves them '''
    def __init__(self, df_train, df_test, target_variable, scaling,
                 results_output_folder,
                 trained_models_output_folder,
                 nb_splits=3, nb_repeats=None,
                 cols_drop=None):

        self.df_train = df_train
        self.df_test = df_test

        if scaling not in ['minmax', 'z-score', 'robust']:
            raise ValueError('Scaling mechanism {} not found. Choose from: {}'.format(scaling, ['minmax', 'z-score', 'robust']))
        # specify scaling mechanism
        if scaling == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling == 'z-score':
            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()

        # drop columns - check if they exist
        self.all_cols = list(df_train.columns)
        if cols_drop:
            # make sure all passed columns exist in the dataset
            for col in cols_drop:
                if col not in self.all_cols:
                    raise ValueError('Column {} you are trying to drop does not exist'.format(col))
            self.df_train = self.df_train.drop(cols_drop, axis=1)
            self.df_test = self.df_test.drop(cols_drop, axis=1)

        # target variable - check if exists
        if target_variable not in self.all_cols:
            raise ValueError('Target variable {} does not exist as a column'.format(target_variable))
        self.target_variable = target_variable

        # train/test input/output features
        self.X_train = np.array(self.df_train.loc[:, self.df_train.columns != self.target_variable])
        self.y_train = np.array(self.df_train.loc[:, self.df_train.columns == self.target_variable]).flatten()

        self.X_test = np.array(self.df_test.loc[:, self.df_test.columns != self.target_variable])
        self.y_test = np.array(self.df_test.loc[:, self.df_test.columns == self.target_variable]).flatten()

        # number of splits and repeats for cross validation
        self.nb_splits = nb_splits
        self.nb_repeats = nb_repeats

        # # (tn, fp, fn, tp)
        self.results_validation = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1', 'auc'])
        self.results_testing = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1', 'auc',
                                                     'tn', 'fp', 'fn', 'tp'])

        # output folder for storing results (error metrics)
        self.results_output_folder = results_output_folder
        self.trained_models_output_folder = trained_models_output_folder
        self.mkdir(self.results_output_folder)
        self.mkdir(self.trained_models_output_folder)

    def cross_validation(self, model, model_name):
        X_train, y_train = self.X_train, self.y_train

        accuracies, precisions, recalls, f_measures, aucs = [], [], [], [], []
        if self.nb_repeats is None:
            kf = StratifiedKFold(n_splits=self.nb_splits)
        else:
            kf = RepeatedStratifiedKFold(n_splits=self.nb_splits, n_repeats=self.nb_repeats)

        # cross validation
        for train_index, test_index in kf.split(X_train, y_train):
            X_train_inner, X_val = X_train[train_index], X_train[test_index]
            y_train_inner, y_val = y_train[train_index], y_train[test_index]

            X_train_inner = self.scaler.fit_transform(X_train_inner)
            X_val = self.scaler.transform(X_val)

            # train and test
            model.fit(X_train_inner, y_train_inner)
            y_pred = model.predict(X_val)

            # store scores
            accuracies.append(accuracy_score(y_val, y_pred))
            precisions.append(precision_score(y_val, y_pred))
            recalls.append(recall_score(y_val, y_pred))
            f_measures.append(f1_score(y_val, y_pred))
            aucs.append(roc_auc_score(y_val, y_pred))

        print('\ncross validation scores:')
        print('accuracy: {:.3f} +- {:.3f}'.format(np.mean(accuracies), np.std(accuracies)))
        print('precision: {:.3f} +- {:.3f}'.format(np.mean(precisions), np.std(precisions)))
        print('recall: {:.3f} +- {:.3f}'.format(np.mean(recalls), np.std(recalls)))
        print('f1: {:.3f} +- {:.3f}'.format(np.mean(f_measures), np.std(f_measures)))
        print('auc: {:.3f} +- {:.3f}'.format(np.mean(aucs), np.std(aucs)))

        self.results_validation.loc[model_name] = pd.Series({
            'accuracy': '{:.3f} +- {:.3f}'.format(np.mean(accuracies), np.std(accuracies)),
            'precision': '{:.3f} +- {:.3f}'.format(np.mean(precisions), np.std(precisions)),
            'recall': '{:.3f} +- {:.3f}'.format(np.mean(recalls), np.std(recalls)),
            'f1': '{:.3f} +- {:.3f}'.format(np.mean(f_measures), np.std(f_measures)),
            'auc': '{:.3f} +- {:.3f}'.format(np.mean(aucs), np.std(aucs)),
        })

        self.results_validation.to_csv(os.path.join(self.results_output_folder, 'errors_validation.csv'))
        print('saved validation error metrics for {}'.format(model_name))

    def train_test_model(self, model, model_name, apply_smote=False):
        X_train, y_train = self.X_train, self.y_train
        X_test, y_test = self.X_test, self.y_test

        # scaling input training and testing
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        if apply_smote:

            # generate smoted training data
            smt = SMOTE(random_state=0)
            X_train_SMOTE, y_train_SMOTE = smt.fit_sample(X_train, y_train)

            # for debugging
            print('Before Smote: {}'.format(Counter(y_train)))
            print('After Smote: {}'.format(Counter(y_train_SMOTE)))

            model.fit(X_train_SMOTE, y_train_SMOTE)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        print('\ntesting scores:')
        accuracy, precision, recall, f1, auc, typei_errors = self.evaluate_predictions(y_test, y_pred)

        # (tn, fp, fn, tp)
        self.results_testing.loc[model_name] = pd.Series({
            'accuracy': '{:.3f}'.format(accuracy),
            'precision': '{:.3f}'.format(precision),
            'recall': '{:.3f}'.format(recall),
            'f1': '{:.3f}'.format(f1),
            'auc': '{:.3f}'.format(auc),
            'tn': typei_errors[0],
            'fp': typei_errors[1],
            'fn': typei_errors[2],
            'tp': typei_errors[3]
        })

        self.results_testing.to_csv(os.path.join(self.results_output_folder, 'errors_testing.csv'))
        print('saved testing error metrics for {}'.format(model_name))

        self.save_trained_model(trained_model=model, trained_model_name=model_name)
        print('saved trained model as {}.sav in {}'.format(model_name, self.trained_models_output_folder))

    def save_trained_model(self, trained_model, trained_model_name):
        ''' saves trained model in specific directory '''
        pickle.dump(trained_model, open(os.path.join(self.trained_models_output_folder, '{}.sav'.format(trained_model_name)),
                                        'wb'))

    def evaluate_predictions(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)

        print('Accuracy: %.3f' % accuracy)
        print('Precision: %.3f' % precision)
        print('Recall: %.3f' % recall)
        print('F1: %.3f' % f1)
        print('AUC: %.3f' % auc)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print('\nConfusion Matrix:')
        print('TN = {}    FP = {}'.format(tn, fp))
        print('FN = {}    TP = {}'.format(fn, tp))

        typei_errors = (tn, fp, fn, tp)

        return accuracy, precision, recall, f1, auc, typei_errors

    def mkdir(self, output_folder):
        ''' create directory if it does not already exist '''
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

