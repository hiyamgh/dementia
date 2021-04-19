import pandas as pd
import numpy as np
import os
import pickle
import itertools
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import category_encoders as ce
from fpgrowth_py import fpgrowth
import warnings

warnings.filterwarnings("ignore")


def encode_categorical_data_supervised(X_train, y_train, X_test, cat_cols, enc_method):
    if enc_method == 'catboost':
        encoder = ce.CatBoostEncoder(cols=cat_cols)
    elif enc_method == 'glmm':
        encoder = ce.GLMMEncoder(cols=cat_cols)
    elif enc_method == 'target':
        encoder = ce.TargetEncoder(cols=cat_cols)
    elif enc_method == 'mestimator':
        encoder = ce.MEstimateEncoder(cols=cat_cols)
    elif enc_method == 'james':
        encoder = ce.JamesSteinEncoder(cols=cat_cols)
    else: # woe
        encoder = ce.WOEEncoder(cols=cat_cols)

    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_test_enc = encoder.transform(X_test)
    print(X_train_enc.shape, X_test_enc.shape)
    return X_train_enc, X_test_enc, encoder


class AdvancedEvaluator:

    def __init__(self, df, df_train, df_test, target_variable,
                 plots_output_folder,
                 fp_growth_output_folder,
                 models_dict,
                 scaling='robust',
                 encodings_dict=None,
                 cat_cols=None,
                 cols_drop=None,
                 pos_class_label=1):

        # main df, train df, test df, and target variable
        self.df = df
        self.df_train = df_train
        self.df_test = df_test
        self.target_variable = target_variable

        # columns to drop -- if any
        # drop columns - check if they exist

        self.all_cols = list(df_test.columns)
        if cols_drop:
            # make sure all passed columns exist in the dataset
            for col in cols_drop:
                if col not in self.all_cols:
                    raise ValueError('Column {} you are trying to drop does not exist'.format(col))

            self.df = self.df.drop(cols_drop, axis=1)
            self.df_train = self.df_train.drop(cols_drop, axis=1)
            self.df_test = self.df_test.drop(cols_drop, axis=1)

        # train/test input/output features
        # we will be using trained models so no need for training data but just in case
        self.X_train = np.array(self.df_train.loc[:, self.df_train.columns != self.target_variable])
        self.y_train = np.array(self.df_train.loc[:, self.df_train.columns == self.target_variable]).flatten()

        self.X_test = np.array(self.df_test.loc[:, self.df_test.columns != self.target_variable])
        self.y_test = np.array(self.df_test.loc[:, self.df_test.columns == self.target_variable]).flatten()
        self.y_test_df = self.df_test.loc[:, self.df_test.columns == self.target_variable]

        self.fprs, self.tprs, self.threshs, self.model_names = [], [], [], []
        self.mean_empirical_risks = []
        self.risk_dfs = []
        self.y_preds = []

        # the label of the positive classes -- usually its 1 but with the exception of fake news data,
        # the positive (fake) is 0
        self.pos_class_label = pos_class_label
        # directory to load trained models from
        # if not os.path.exists(trained_models_dir):
        #     raise ValueError('Directory of trained models: {} does not exist'.format(trained_models_dir))
        # self.trained_models_dir = trained_models_dir

        # directory for dumping output plots
        self.mkdir(output_folder=plots_output_folder)
        self.mkdir(output_folder=fp_growth_output_folder)
        self.plots_output_folder = plots_output_folder
        self.fp_growth_output_folder = fp_growth_output_folder

        # get models dictionary
        self.models_dict = models_dict
        self.scaling = scaling

        # scale the data, if need be
        if self.scaling is not None:
            if scaling not in ['minmax', 'z-score', 'robust']:
                raise ValueError(
                    'Scaling mechanism {} not found. Choose from: {}'.format(scaling, ['minmax', 'z-score', 'robust']))
            if scaling == 'minmax':
                self.scaler = MinMaxScaler()
            elif scaling == 'z-score':
                self.scaler = StandardScaler()
            else:
                self.scaler = RobustScaler()

        # Encode categorical columns in the data, if need be
        self.encodings_dict = encodings_dict # categorical encoding mechanism
        self.cat_cols = cat_cols # categorical columns

    def mkdir(self, output_folder):
        ''' create directory if it does not already exist '''
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def classify(self, trained_model, trained_model_name, nb_bins=10):
        if self.encodings_dict is not None:
            X_train_df = self.df_train.drop([self.target_variable], axis=1)
            y_train_df = self.df_train[[self.target_variable]]
            X_test_df = self.df_test.drop([self.target_variable], axis=1)
            X_train_enc, X_test_enc, encoder = encode_categorical_data_supervised(X_train=X_train_df,
                                                                                  y_train=y_train_df,
                                                                                  X_test=X_test_df,
                                                                                  cat_cols=self.cat_cols,
                                                                                  enc_method=self.encodings_dict[trained_model_name])

            self.X_train = np.array(X_train_enc)
            self.X_test = np.array(X_test_enc)
            self.encoder = encoder

        X_train, y_train, X_test, y_test = self.X_train, self.y_train, self.X_test, self.y_test

        if self.scaling is not None:
            # scale the data so that you can predict on scaled testing data directly
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        # number of bins for generating mean empirical risk curves
        self.nb_bins = nb_bins

        # WILL NOT TRAIN -- MODEL ALREADY TRAINED
        # model.fit(X_train, y_train)

        # predict classes/labels
        y_pred = trained_model.predict(X_test)
        if self.pos_class_label == 1:
            probas = trained_model.predict_proba(X_test)[:, 1]
        else:
            probas = trained_model.predict_proba(X_test)[:, 0]
        self.evaluate_predictions(y_test=y_test, y_pred=y_pred)
        self.y_preds.append(y_pred)

        # predict probabilities for ROC Curves
        fpr, tpr, thresh = roc_curve(self.y_test, probas, pos_label=1)
        self.fprs.append(fpr)
        self.tprs.append(tpr)
        self.threshs.append(thresh)
        self.model_names.append(trained_model_name)

        # get topK% at risk
        # get the indices of each testing instance along with its risk score
        test_indexes = list(self.y_test_df.index)
        self.test_indexes = test_indexes
        risk_scores = probas

        # create a new dataframe of indices & their risk
        risk_df = pd.DataFrame(np.column_stack((test_indexes, y_test, y_pred, risk_scores)),
                               columns=['test_indices', 'y_test', 'y_pred', 'risk_scores'])
        risk_df = risk_df.sort_values(by='risk_scores', ascending=False)

        # # # create 4 bins of the data (like in the paper)
        # # risk_df['quantiles'] = pd.qcut(risk_df['risk_scores'], q=4, duplicates='drop')
        # risk_df['quantiles'] = pd.cut(risk_df['risk_scores'], self.nb_bins)
        # print(pd.cut(risk_df['risk_scores'], self.nb_bins).value_counts())
        items_per_bin = len(risk_df) // self.nb_bins
        bin_category = [0] * len(risk_df)
        for i in range(self.nb_bins):
            lower = i * items_per_bin
            if i != self.nb_bins - 1:
                upper = (i + 1) * items_per_bin
                bin_category[lower:upper] = [i] * (upper - lower)
            else:
                bin_category[lower:] = [i] * (len(range(lower, len(risk_df))))

        risk_df['quantiles'] = list(reversed(bin_category))
        # calculate mean empirical risk
        # which is the fraction of students from that bin who actually (as per ground truth)
        # failed to graduate on time (actually positive)
        mean_empirical_risk = []
        quantiles_sorted = sorted(list(risk_df['quantiles'].unique()))
        for quantile in quantiles_sorted:
            df = risk_df[risk_df['quantiles'] == quantile]
            # test_indices_curr = df['test_indices']
            # ground_truth = self.y_test_df.loc[test_indices_curr, :]
            ground_truth = df['y_test']
            # mean_empirical_risk.append(list(ground_truth[self.target_variable]).count(1)/len(ground_truth))
            mean_empirical_risk.append(list(ground_truth).count(1) / len(ground_truth))
        print('quantiles: {}'.format(quantiles_sorted))
        print('mean empirical risk: {}'.format(mean_empirical_risk))
        self.mean_empirical_risks.append(mean_empirical_risk)

        self.risk_dfs.append(risk_df)

    def compute_jaccard_similarity(self, topKs):
        ''' Jaccard Similarity at Top K '''
        combinations = list(itertools.combinations(list(self.models_dict.keys()), 2))
        models_indexes = {}

        for index, (key, value) in enumerate(self.models_dict.items()):
            models_indexes[key] = index

        for combn in combinations:
            # get the two model pairs
            clf1 = combn[0]
            clf2 = combn[1]

            # get their indexes
            indx1 = models_indexes[clf1]
            indx2 = models_indexes[clf2]

            # get each model's risk_df
            r1 = self.risk_dfs[indx1]
            r2 = self.risk_dfs[indx2]

            jaccard_sims = []
            for topk in topKs:
                r1_curr = r1.head(n=topk)
                r2_curr = r2.head(n=topk)

                test_indices1 = list(r1_curr['test_indices'])
                test_indices2 = list(r2_curr['test_indices'])
                jaccard_sim = len(set(test_indices1).intersection(set(test_indices2))) / len(
                    set(test_indices1).union(set(test_indices2)))
                jaccard_sims.append(jaccard_sim)

            plt.plot(topKs, jaccard_sims, marker='o', label='{}-{}'.format(clf1, clf2))

        plt.legend(loc='best')
        plt.xlabel('Top K')

        plt.ylabel('Jaccard Similarity')
        plt.savefig(os.path.join(self.plots_output_folder, 'jaccard_topK.png'))
        plt.close()

    def evaluate_predictions(self, y_test, y_pred):
        ''' Evaluate predictions using standard error metrics like accuracy, precision, recall, F1, AUC '''
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        print('Accuracy: %.3f' % accuracy)
        print('Precision: %.3f' % precision)
        print('Recall: %.3f' % recall)
        print('F1: %.3f' % f1)
        print('AUC: %.3f' % auc)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print('\nConfusion Matrix:')
        print('TN = {}    FP = {}'.format(tn, fp))
        print('FN = {}    TP = {}'.format(fn, tp))

    def produce_roc_curves(self):
        ''' ROC AUC Curves '''
        # plot roc curves
        for i in range(len(self.fprs)):
            plt.plot(self.fprs[i], self.tprs[i], linestyle='--', label=self.model_names[i])
        # title
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        # plt.savefig('roc_curves.png')
        plt.savefig(os.path.join(self.plots_output_folder, 'roc_curves.png'))
        plt.close()

    def produce_empirical_risk_curves(self):
        ''' produce plot of mean empirical risks '''
        for i in range(len(self.models_dict)):
            # xaxis = list(range(len(self.mean_empirical_risks[i])))
            plt.plot(range(1, self.nb_bins + 1), self.mean_empirical_risks[i], marker='o', label=self.model_names[i])
        plt.legend(loc='best')
        plt.xlabel('Bins')
        plt.ylabel('Mean Empirical risks')
        plt.savefig(os.path.join(self.plots_output_folder, 'mean_empirical_risks.png'))
        plt.close()

    def produce_curves_topK(self, topKs, metric):
        ''' precision & recall at top K curves '''
        fig, ax = plt.subplots()
        for i in range(len(self.models_dict)):
            risk_df = self.risk_dfs[i]
            metrics = []
            for topk in topKs:
                risk_df_curr = risk_df.head(n=topk)
                # test_indices = list(risk_df_curr['test_indices'])
                y_pred_curr = list(risk_df_curr['y_pred'])
                # y_true_curr = list(self.y_test_df.loc[test_indices, self.target_variable])
                y_true_curr = list(risk_df_curr['y_test'])
                if metric == 'precision':
                    precision_curr = precision_score(y_true_curr, y_pred_curr)
                    metrics.append(precision_curr)
                else:
                    recall_curr = recall_score(y_true_curr, y_pred_curr)
                    metrics.append(recall_curr)
            plt.plot(topKs, metrics, label=self.model_names[i], marker='o')

        plt.legend(loc='best')
        plt.xlabel('Top K')
        # ax.set_xlim(ax.get_xlim()[::-1])
        if metric == 'precision':
            plt.ylabel('Precision')
            plt.savefig(os.path.join(self.plots_output_folder, 'precisions_topK.png'))
        else:
            plt.ylabel('Recall')
            plt.savefig(os.path.join(self.plots_output_folder, 'recalls_topK.png'))
        plt.close()

    def identify_frequent_patterns(self):
        ''' * identify frequent patterns in the data
            * Assume all inputs are numeric
            :return
                * FreqItemSet: List of all frequent patterns in the data
                * cols_meta: Dictionary of meta data of each column in the data
                    (i.e. quantile values of the distribution of each column)
        '''

        # inputs needed
        df = self.df
        itemSetList = []
        df_cols = list(df.columns)
        df_cols.remove(self.target_variable)

        #  get the 25th, 50th, and 75th quartiles of each column
        self.cols_meta = {}
        for col in df_cols:
            self.cols_meta[col] = {
                'min': df[col].min(),
                '25th': df[col].quantile(0.25),
                '50th': df[col].quantile(0.50),
                '75th': df[col].quantile(0.75),
                'max': df[col].max()
            }
            keys_to_delete = []
            keys = list(self.cols_meta[col].keys())
            vals = list(self.cols_meta[col].values())
            for i in range(len(keys) - 1):
                if vals[i] == vals[i + 1]:
                    keys_to_delete.append(keys[i + 1])

            # delete keys
            if keys_to_delete:
                for k in keys_to_delete:
                    del self.cols_meta[col][k]

        # use these quantiles for categorizing data
        for index, row in df.iterrows():
            curr_items = []
            for col in df_cols:

                # if self.cols_meta[col]['min'] <= row[col] < self.cols_meta[col]['25th']:
                #     curr_items.append('{:.2f}<{}<{:.2f}'.format(self.cols_meta[col]['min'], col, self.cols_meta[col]['25th']))
                #
                # elif self.cols_meta[col]['25th'] <= row[col] < self.cols_meta[col]['50th']:
                #     curr_items.append('{:.2f}<{}<{:.2f}'.format(self.cols_meta[col]['25th'], col, self.cols_meta[col]['50th']))
                #
                # elif self.cols_meta[col]['50th'] <= row[col] < self.cols_meta[col]['75th']:
                #     curr_items.append('{:.2f}<{}<{:.2f}'.format(self.cols_meta[col]['50th'], col, self.cols_meta[col]['75th']))
                #
                # else:
                #     curr_items.append('{:.2f}<{}<{:.2f}'.format(self.cols_meta[col]['75th'], col, self.cols_meta[col]['max']))

                percentiles = list(self.cols_meta[col].keys())
                percentiles_pairs = list(zip(percentiles, percentiles[1:]))
                for pair in percentiles_pairs:
                    if pair[1] != 'max':
                        if self.cols_meta[col][pair[0]] <= row[col] < self.cols_meta[col][pair[1]]:
                            curr_items.append(
                                '{}<{}<{}'.format(self.cols_meta[col][pair[0]], col, self.cols_meta[col][pair[1]]))
                            break
                    else:
                        curr_items.append(
                            '{}<{}<{}'.format(self.cols_meta[col][pair[0]], col, self.cols_meta[col][pair[1]]))

            itemSetList.append(curr_items)

        # get the frequent patterns
        self.freqItemSet, self.rules = fpgrowth(itemSetList, minSupRatio=0.5, minConf=0.5)
        fps = ['fp{}'.format(i) for i in range(1, len(self.freqItemSet) + 1)]

        # create a dictionary of frequent patterns
        self.fp_dict = dict(zip(fps, list(self.freqItemSet)))

        # write the frequent patterns to a txt file
        with open(os.path.join(self.fp_growth_output_folder, 'frequent_patterns.txt'), 'w') as txt_file:
            for fp in self.freqItemSet:
                txt_file.write(str(fp) + '\n')

    def add_mistake(self):
        ''' adds the mistake next to each prediction '''
        risk_dfs_updated = []
        for index, risk_df in enumerate(self.risk_dfs):
            risk_df['mistake'] = risk_df.apply(lambda row: 0 if row['y_test'] == row['y_pred'] else 1, axis=1)
            risk_df = risk_df[['test_indices', 'y_test', 'y_pred', 'risk_scores', 'mistake', 'quantiles']]
            risk_dfs_updated.append(risk_df)

        self.risk_dfs = risk_dfs_updated
        # self.pattern_probability_of_mistake()

    def fp_in_df(self, fp, df):

        ''' check if the frequent pattern is in the passed df -- return a df
            sub-setted by the rows that include the passed fp
        '''

        def get_bounds(col, lower, upper):
            main_dict = self.cols_meta[col]
            # # min-25th
            # if main_dict['min'] == float(lower) and main_dict['25th'] == float(upper):
            #     return ['min', '25th']
            # elif main_dict['25th'] == float(lower) and main_dict['50th'] == float(upper):
            #     return ['25th', '50th']
            # elif main_dict['50th'] == float(lower) and main_dict['75th'] == float(upper):
            #     return ['50th', '75th']
            # else:
            #     return ['75th', 'max']
            percentiles = list(main_dict.keys())
            percentiles_pairs = list(zip(percentiles, percentiles[1:]))
            for pair in percentiles_pairs:
                if main_dict[pair[0]] == float(lower) and main_dict[pair[1]] == float(upper):
                    return [pair[0], pair[1]]

        fps = list(self.fp_dict[fp])
        col_names, lower_bounds, upper_bounds = [], [], []
        for fp in fps:
            col = fp.split('<')[1]
            lower = fp.split('<')[0]
            upper = fp.split('<')[2]

            lb, ub = get_bounds(col, lower, upper)

            # add to the list of current fps
            col_names.append(col)
            lower_bounds.append(lb)
            upper_bounds.append(ub)

        indices_who_has_fp = []
        for index, row in df.iterrows():
            add_index = True
            for i, col in enumerate(col_names):
                valuec = row[col]  # current value
                lbc, ubc = lower_bounds[i], upper_bounds[i]  # lower bound current, upper bound current
                if lbc == '75th':
                    if self.cols_meta[col][lbc] <= valuec <= self.cols_meta[col][ubc]:
                        pass
                    else:
                        add_index = False
                        break
                else:
                    if self.cols_meta[col][lbc] <= valuec < self.cols_meta[col][ubc]:
                        pass
                    else:
                        add_index = False
                        break
            if add_index:
                indices_who_has_fp.append(index)

        return df[df.index.isin(indices_who_has_fp)]

    def pattern_probability_of_mistake(self):
        # self.create_freq_pattern_dict()
        probabilities_per_fp = {}
        for index, model in enumerate(self.models_dict):
            probabilities_per_fp[model] = {}
            # for freq_pattern in self.freq_patterns:
            # for freq_pattern in self.freqItemSet:
            for freq_pattern in self.fp_dict:

                df_test = self.df[self.df.index.isin(self.test_indexes)]
                df_test_fp = self.fp_in_df(freq_pattern, df_test)
                indices_fp = list(df_test_fp.index)

                if not indices_fp:
                    probabilities_per_fp[model][freq_pattern] = -1
                else:
                    risk_df = self.risk_dfs[index]
                    risk_df = risk_df[risk_df['test_indices'].isin(indices_fp)]
                    prob_mistake = len(risk_df[risk_df['mistake'] == 1]) / len(risk_df)
                    # if not probabilities_per_fp[model]:
                    probabilities_per_fp[model][freq_pattern] = prob_mistake

        df_probas = pd.DataFrame()
        df_probas['FP'] = [v for k, v in self.fp_dict.items()]
        for model in self.models_dict:
            df_probas[model] = [v for k, v in probabilities_per_fp[model].items()]

        self.mkdir(output_folder=self.fp_growth_output_folder)
        df_probas.to_csv(os.path.join(self.fp_growth_output_folder, 'fp_growth.csv'), index=False)

        return probabilities_per_fp
