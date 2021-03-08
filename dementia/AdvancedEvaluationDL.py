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

    def __init__(self, models_results, plots_output_folder, nb_bins=10):

        self.models_results = {}
        for metric in models_results:
            for topn_model in models_results[metric]['topn_models']:
                self.models_results[topn_model] = models_results[metric][topn_model]

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