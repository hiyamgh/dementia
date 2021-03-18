'''
author: Hiyam K. Ghannam
email: hkg02@mail.aub.edu
'''

import pandas as pd
import os
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import *
import warnings
warnings.filterwarnings("ignore")


def mkdir(folder):
    ''' create a directory of specified path if it does not already exist '''
    if not os.path.exists(folder):
        os.makedirs(folder)


class AdvancedEvaluator:

    def __init__(self, models_results, plots_output_folder, fps, df_test, nb_bins=10, special_cases=None):
        '''
        Main class for running Advanced ML Evaluation, inspired by: https://dl.acm.org/doi/10.1145/2783258.2788620
        :param models_results: the risk dfs attached per maml model
        :param plots_output_folder: the name of the output folder
        :param fps: the collection of frequent patterns, one per minimum support
        :param df_test: the testing dataset
        :param nb_bins: number of bins for mean empirical risk curves, be default 10
        :param special_cases: special models of interest, by default, None.
        '''

        self.fps = fps # list of frequent patterns
        self.df_test = df_test
        self.models_results = {}
        if special_cases is None:
            for metric in models_results:
                for topn_model in models_results[metric]['topn_models']:
                    self.models_results[topn_model] = models_results[metric][topn_model]
        else:
            metric = 'f2'
            for sc in special_cases:
                self.models_results[sc] = models_results[metric][sc]

        # self.models_results = models_results
        self.nb_bins = nb_bins

        # directory for dumping output plots
        self.mkdir(output_folder=plots_output_folder)
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
        for model_name_num in self.models_results:
            # already sorted
            risk_df = self.models_results[model_name_num]['risk_df']

            # for producing AUC-ROC Curves
            fpr, tpr, thresh = roc_curve(risk_df['y_test'], risk_df['risk_scores'], pos_label=1)
            self.fprs.append(fpr)
            self.tprs.append(tpr)
            self.threshs.append(thresh)
            # self.model_names.append('model{}'.format(model_name_num))
            self.model_names.append(model_name_num)

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
            for model_num_name in self.models_results:
                risk_df = self.models_results[model_num_name]['risk_df']
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
                # plt.plot(topKs, metrics, label=self.model_names[i-1], marker='o')
                plt.plot(topKs, metrics, label=model_num_name, marker='o')

            plt.legend(loc='best')
            plt.xlabel('Top K')
            if metric == 'precision':
                plt.ylabel('Precision')
                plt.savefig(os.path.join(self.plots_output_folder, 'precisions_topK.png'))
            else:
                plt.ylabel('Recall')
                plt.savefig(os.path.join(self.plots_output_folder, 'recalls_topK.png'))
            plt.close()

    def _create_mistake(self, row):
        if row['y_test'] == row['y_pred']:
            return 0
        return 1
    
    def _get_row_indices(self, fp, df_test):
        indexes = []
        # lower bound, column/feature name, upper bound
        lb, col_name, ub = float(fp.split('<')[0]), fp.split('<')[1], float(fp.split('<')[2])
        for i, row in df_test.iterrows():
            if lb <= row[col_name] <= ub:
                indexes.append(i)
        
        return indexes
        
    def characterize_prediction_mistakes(self, out_folder):
        for model_name_num in self.models_results:
            # already sorted
            risk_df = self.models_results[model_name_num]['risk_df']
            # create a new field called mistake, set this field to 1 for those data points
            # where the prediction of the classification model does not match ground truth,
            # else, set this field to zero
            risk_df['mistake'] = risk_df.apply(self._create_mistake, axis=1)
            # for each frequent pattern, generate probability of mistake, this can
            # be done by iterating over all data points where the pattern holds true and
            # computing the fraction of these data points where the mistake field is
            # set to 1
            fp_prob_mistakes = pd.DataFrame(columns=['frequent_pattern', 'prob_of_mistake'])
            mkdir(out_folder)
            for fp in self.fps:
                # a frequent pattern may have more than one rule, so:
                mistakesare1 = []
                for r in fp:
                    # get the indices of rows that have the rule/fp (the rows in testing dataset, of course)
                    indexes = self._get_row_indices(r, df_test=self.df_test)
                    # get those indices that have mistake = 1
                    mistakeis1 = [int(row['test_indices']) for _, row in risk_df.iterrows() if row['mistake'] == 1 and int(row['test_indices']) in indexes]
                    mistakesare1.append(mistakeis1)

                results_union = set().union(*mistakesare1)
                # but the risk_df has repeated testing indices due to query_set repetitions (num_updates)
                # //TODO check which is the appropriate 'len' to divide by for getting the probability of mistake
                prob_of_mistake = len(results_union) / len(risk_df) * 100
                fp_prob_mistakes = fp_prob_mistakes.append({
                    'frequent_pattern': fp,
                    'prob_of_mistake':  '{:.2f}'.format(prob_of_mistake)
                }, ignore_index=True)

            fp_prob_mistakes = fp_prob_mistakes.sort_values(by='prob_of_mistakes')
            fp_prob_mistakes.to_csv(os.path.join(out_folder, '{}_prob_mistakes.csv').format(model_name_num))

                    



