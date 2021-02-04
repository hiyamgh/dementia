import pandas as pd
import numpy as np
import os
import pickle
import itertools
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import *
from fpgrowth_py import fpgrowth
import warnings

warnings.filterwarnings("ignore")


class AdvancedEvaluator:

    def __init__(self, models_results, plots_output_folder, fp_growth_output_folder, nb_bins=10):

        self.models_results = models_results
        self.nb_bins = nb_bins

        # directory for dumping output plots
        self.mkdir(output_folder=plots_output_folder)
        self.mkdir(output_folder=fp_growth_output_folder)
        self.plots_output_folder = plots_output_folder
        # self.fp_growth_output_folder = fp_growth_output_folder

        # initialize mean empirical list of lists
        self.mean_empirical_risks = []
        self.fprs, self.tprs, self.threshs, self.model_names = [], [], [], []

    def mkdir(self, output_folder):
        ''' create directory if it does not already exist '''
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def build_mean_empirical_risks(self):
        for model_num in self.models_results:
            # already sorted
            risk_df = self.models_results[model_num]['risk_df']

            # for producing AUC-ROC Curves
            fpr, tpr, thresh = roc_curve(risk_df['y_test'], risk_df['risk_scores'], pos_label=1)
            self.fprs.append(fpr)
            self.tprs.append(tpr)
            self.threshs.append(thresh)
            self.model_names.append('model{}'.format(model_num))

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
            mean_empirical_risk = []
            quantiles_sorted = sorted(list(risk_df['quantiles'].unique()))
            for quantile in quantiles_sorted:
                df = risk_df[risk_df['quantiles'] == quantile]
                ground_truth = df['y_test']
                mean_empirical_risk.append(list(ground_truth).count(1) / len(ground_truth))
            print('quantiles: {}'.format(quantiles_sorted))
            print('mean empirical risk: {}'.format(mean_empirical_risk))
            self.mean_empirical_risks.append(mean_empirical_risk)

    def produce_empirical_risk_curves(self):
        ''' produce plot of mean empirical risks '''
        self.build_mean_empirical_risks()
        for i in range(len(self.models_results)):
            plt.plot(range(1, self.nb_bins + 1), self.mean_empirical_risks[i], marker='o', label=self.model_names[i])
        plt.legend(loc='best')
        plt.xlabel('Bins')
        plt.ylabel('Mean Empirical risks')
        plt.savefig(os.path.join(self.plots_output_folder, 'mean_empirical_risks.png'))
        plt.close()

    def compute_jaccard_similarity(self, topKs):
        ''' Jaccard Similarity at Top K '''
        combinations = list(itertools.combinations(list(self.models_results.keys()), 2))

        for combn in combinations:
            # get the two model pairs
            clf1 = combn[0]
            clf2 = combn[1]

            r1 = self.models_results[clf1]['risk_df']
            r2 = self.models_results[clf2]['risk_df']

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

    def produce_roc_curves(self):
        ''' ROC AUC Curves '''
        for i in range(len(self.fprs)):
            plt.plot(self.fprs[i], self.tprs[i], linestyle='--', label=self.model_names[i])
        # title
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.plots_output_folder, 'roc_curves.png'))
        plt.close()

    def produce_curves_topK(self, topKs):
        ''' precision & recall at top K curves '''
        for metric in ['precision', 'recall']:
            for i in self.models_results:
                risk_df = self.models_results[i]['risk_df']
                metrics = []
                for topk in topKs:
                    risk_df_curr = risk_df.head(n=topk)
                    y_pred_curr = list(risk_df_curr['y_pred'])
                    y_true_curr = list(risk_df_curr['y_test'])
                    if metric == 'precision':
                        precision_curr = precision_score(y_true_curr, y_pred_curr)
                        metrics.append(precision_curr)
                    else:
                        recall_curr = recall_score(y_true_curr, y_pred_curr)
                        metrics.append(recall_curr)
                plt.plot(topKs, metrics, label=self.model_names[i-1], marker='o')

            plt.legend(loc='best')
            plt.xlabel('Top K')
            if metric == 'precision':
                plt.ylabel('Precision')
                plt.savefig(os.path.join(self.plots_output_folder, 'precisions_topK.png'))
            else:
                plt.ylabel('Recall')
                plt.savefig(os.path.join(self.plots_output_folder, 'recalls_topK.png'))
            plt.close()

    # def identify_frequent_patterns(self):
    #     ''' * identify frequent patterns in the data
    #         * Assume all inputs are numeric
    #         :return
    #             * FreqItemSet: List of all frequent patterns in the data
    #             * cols_meta: Dictionary of meta data of each column in the data
    #                 (i.e. quantile values of the distribution of each column)
    #     '''
    #
    #     # inputs needed
    #     df = self.df
    #     itemSetList = []
    #     df_cols = list(df.columns)
    #     df_cols.remove(self.target_variable)
    #
    #     #  get the 25th, 50th, and 75th quartiles of each column
    #     self.cols_meta = {}
    #     for col in df_cols:
    #         self.cols_meta[col] = {
    #             'min': df[col].min(),
    #             '25th': df[col].quantile(0.25),
    #             '50th': df[col].quantile(0.50),
    #             '75th': df[col].quantile(0.75),
    #             'max': df[col].max()
    #         }
    #         keys_to_delete = []
    #         keys = list(self.cols_meta[col].keys())
    #         vals = list(self.cols_meta[col].values())
    #         for i in range(len(keys) - 1):
    #             if vals[i] == vals[i + 1]:
    #                 keys_to_delete.append(keys[i + 1])
    #
    #         # delete keys
    #         if keys_to_delete:
    #             for k in keys_to_delete:
    #                 del self.cols_meta[col][k]
    #
    #     # use these quantiles for categorizing data
    #     for index, row in df.iterrows():
    #         curr_items = []
    #         for col in df_cols:
    #
    #             # if self.cols_meta[col]['min'] <= row[col] < self.cols_meta[col]['25th']:
    #             #     curr_items.append('{:.2f}<{}<{:.2f}'.format(self.cols_meta[col]['min'], col, self.cols_meta[col]['25th']))
    #             #
    #             # elif self.cols_meta[col]['25th'] <= row[col] < self.cols_meta[col]['50th']:
    #             #     curr_items.append('{:.2f}<{}<{:.2f}'.format(self.cols_meta[col]['25th'], col, self.cols_meta[col]['50th']))
    #             #
    #             # elif self.cols_meta[col]['50th'] <= row[col] < self.cols_meta[col]['75th']:
    #             #     curr_items.append('{:.2f}<{}<{:.2f}'.format(self.cols_meta[col]['50th'], col, self.cols_meta[col]['75th']))
    #             #
    #             # else:
    #             #     curr_items.append('{:.2f}<{}<{:.2f}'.format(self.cols_meta[col]['75th'], col, self.cols_meta[col]['max']))
    #
    #             percentiles = list(self.cols_meta[col].keys())
    #             percentiles_pairs = list(zip(percentiles, percentiles[1:]))
    #             for pair in percentiles_pairs:
    #                 if pair[1] != 'max':
    #                     if self.cols_meta[col][pair[0]] <= row[col] < self.cols_meta[col][pair[1]]:
    #                         curr_items.append(
    #                             '{}<{}<{}'.format(self.cols_meta[col][pair[0]], col, self.cols_meta[col][pair[1]]))
    #                         break
    #                 else:
    #                     curr_items.append(
    #                         '{}<{}<{}'.format(self.cols_meta[col][pair[0]], col, self.cols_meta[col][pair[1]]))
    #
    #         itemSetList.append(curr_items)
    #
    #     # get the frequent patterns
    #     self.freqItemSet, self.rules = fpgrowth(itemSetList, minSupRatio=0.5, minConf=0.5)
    #     fps = ['fp{}'.format(i) for i in range(1, len(self.freqItemSet) + 1)]
    #
    #     # create a dictionary of frequent patterns
    #     self.fp_dict = dict(zip(fps, list(self.freqItemSet)))
    #
    #     # write the frequent patterns to a txt file
    #     with open(os.path.join(self.fp_growth_output_folder, 'frequent_patterns.txt'), 'w') as txt_file:
    #         for fp in self.freqItemSet:
    #             txt_file.write(str(fp) + '\n')
    #
    # def add_mistake(self):
    #     ''' adds the mistake next to each prediction '''
    #     risk_dfs_updated = []
    #     for index, risk_df in enumerate(self.risk_dfs):
    #         risk_df['mistake'] = risk_df.apply(lambda row: 0 if row['y_test'] == row['y_pred'] else 1, axis=1)
    #         risk_df = risk_df[['test_indices', 'y_test', 'y_pred', 'risk_scores', 'mistake', 'quantiles']]
    #         risk_dfs_updated.append(risk_df)
    #
    #     self.risk_dfs = risk_dfs_updated
    #     # self.pattern_probability_of_mistake()
    #
    # def fp_in_df(self, fp, df):
    #
    #     ''' check if the frequent pattern is in the passed df -- return a df
    #         sub-setted by the rows that include the passed fp
    #     '''
    #
    #     def get_bounds(col, lower, upper):
    #         main_dict = self.cols_meta[col]
    #         # # min-25th
    #         # if main_dict['min'] == float(lower) and main_dict['25th'] == float(upper):
    #         #     return ['min', '25th']
    #         # elif main_dict['25th'] == float(lower) and main_dict['50th'] == float(upper):
    #         #     return ['25th', '50th']
    #         # elif main_dict['50th'] == float(lower) and main_dict['75th'] == float(upper):
    #         #     return ['50th', '75th']
    #         # else:
    #         #     return ['75th', 'max']
    #         percentiles = list(main_dict.keys())
    #         percentiles_pairs = list(zip(percentiles, percentiles[1:]))
    #         for pair in percentiles_pairs:
    #             if main_dict[pair[0]] == float(lower) and main_dict[pair[1]] == float(upper):
    #                 return [pair[0], pair[1]]
    #
    #     fps = list(self.fp_dict[fp])
    #     col_names, lower_bounds, upper_bounds = [], [], []
    #     for fp in fps:
    #         col = fp.split('<')[1]
    #         lower = fp.split('<')[0]
    #         upper = fp.split('<')[2]
    #
    #         lb, ub = get_bounds(col, lower, upper)
    #
    #         # add to the list of current fps
    #         col_names.append(col)
    #         lower_bounds.append(lb)
    #         upper_bounds.append(ub)
    #
    #     indices_who_has_fp = []
    #     for index, row in df.iterrows():
    #         add_index = True
    #         for i, col in enumerate(col_names):
    #             valuec = row[col]  # current value
    #             lbc, ubc = lower_bounds[i], upper_bounds[i]  # lower bound current, upper bound current
    #             if lbc == '75th':
    #                 if self.cols_meta[col][lbc] <= valuec <= self.cols_meta[col][ubc]:
    #                     pass
    #                 else:
    #                     add_index = False
    #                     break
    #             else:
    #                 if self.cols_meta[col][lbc] <= valuec < self.cols_meta[col][ubc]:
    #                     pass
    #                 else:
    #                     add_index = False
    #                     break
    #         if add_index:
    #             indices_who_has_fp.append(index)
    #
    #     return df[df.index.isin(indices_who_has_fp)]
    #
    # def pattern_probability_of_mistake(self):
    #     # self.create_freq_pattern_dict()
    #     probabilities_per_fp = {}
    #     for index, model in enumerate(self.models_dict):
    #         probabilities_per_fp[model] = {}
    #         # for freq_pattern in self.freq_patterns:
    #         # for freq_pattern in self.freqItemSet:
    #         for freq_pattern in self.fp_dict:
    #
    #             df_test = self.df[self.df.index.isin(self.test_indexes)]
    #             df_test_fp = self.fp_in_df(freq_pattern, df_test)
    #             indices_fp = list(df_test_fp.index)
    #
    #             if not indices_fp:
    #                 probabilities_per_fp[model][freq_pattern] = -1
    #             else:
    #                 risk_df = self.risk_dfs[index]
    #                 risk_df = risk_df[risk_df['test_indices'].isin(indices_fp)]
    #                 prob_mistake = len(risk_df[risk_df['mistake'] == 1]) / len(risk_df)
    #                 # if not probabilities_per_fp[model]:
    #                 probabilities_per_fp[model][freq_pattern] = prob_mistake
    #
    #     df_probas = pd.DataFrame()
    #     df_probas['FP'] = [v for k, v in self.fp_dict.items()]
    #     for model in self.models_dict:
    #         df_probas[model] = [v for k, v in probabilities_per_fp[model].items()]
    #
    #     self.mkdir(output_folder=self.fp_growth_output_folder)
    #     df_probas.to_csv(os.path.join(self.fp_growth_output_folder, 'fp_growth.csv'), index=False)
    #
    #     return probabilities_per_fp
