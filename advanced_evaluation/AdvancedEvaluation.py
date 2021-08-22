import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt
from sklearn.metrics import *
import warnings

warnings.filterwarnings("ignore")


class AdvancedEvaluator:

    def __init__(self, plots_output_folder, models_results, nb_bins=10):

        self.models_results = models_results

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

    def build_mean_empirical_risks(self, topn=None):
        self.risk_dfs = []
        for model_name_num in self.models_results:
            # already sorted
            if topn:
                risk_df = pd.read_csv(self.models_results[model_name_num]).head(topn)
            else:
                risk_df = pd.read_csv(self.models_results[model_name_num])
            self.risk_dfs.append(risk_df)

            # for producing AUC-ROC Curves
            fpr, tpr, thresh = roc_curve(risk_df['y_test'], risk_df['risk_scores'])
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
                # mean_empirical_risk.append(list(ground_truth).count(1) / len(ground_truth))
                mean_empirical_risk.append(list(ground_truth).count(1) / len(ground_truth))
            print('quantiles: {}'.format(quantiles_sorted))
            print('mean empirical risk: {}'.format(mean_empirical_risk))
            print('len: {}'.format(len(mean_empirical_risk)))
            self.mean_empirical_risks.append(mean_empirical_risk)

    def produce_empirical_risk_curves(self, topn=None):
        ''' produce plot of mean empirical risks '''
        self.build_mean_empirical_risks(topn=topn)
        for i in range(len(self.models_results)):
            plt.plot(range(1, self.nb_bins + 1), self.mean_empirical_risks[i], marker='o', label=self.model_names[i])
        plt.legend(loc='best')
        plt.xlabel('Bins')
        plt.ylabel('Mean Empirical risks')
        plt.savefig(os.path.join(self.plots_output_folder, 'mean_empirical_risks.png'))
        plt.savefig(os.path.join(self.plots_output_folder, 'mean_empirical_risks.pdf'), dpi=300)
        plt.close()

    def produce_roc_curves(self):
        ''' ROC AUC Curves '''
        for i in range(len(self.fprs)):
            plt.plot(self.fprs[i], self.tprs[i], linestyle='--', label=self.model_names[i])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.plots_output_folder, 'roc_curves.png'))
        plt.savefig(os.path.join(self.plots_output_folder, 'roc_curves.pdf'), dpi=300)
        plt.close()

    def produce_curves_topK(self, topKs):
        ''' precision & recall at top K curves '''
        for metric in ['precision', 'recall']:
            i = 0
            for model_num_name in self.models_results:
                risk_df = self.risk_dfs[i]
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
                plt.ylim([0.0, 1.1])
                i += 1

            plt.legend(loc='best')
            plt.xlabel('Top K')
            if metric == 'precision':
                plt.ylabel('Precision')
                plt.savefig(os.path.join(self.plots_output_folder, 'precisions_topK.png'))
                plt.savefig(os.path.join(self.plots_output_folder, 'precisions_topK.pdf'), dpi=300)
            else:
                plt.ylabel('Recall')
                plt.savefig(os.path.join(self.plots_output_folder, 'recalls_topK.png'))
                plt.savefig(os.path.join(self.plots_output_folder, 'recalls_topK.pdf'), dpi=300)
            plt.close()


if __name__ == '__main__':

    models_results = {
        'MAML': 'risk_dataframes/MAML - top20.csv',
        # 'ARML-top20': 'arml/dm_top20/model_39/risk_df.csv',
        'Reptile': 'risk_dataframes/Reptile - top20.csv',
        'Reptile Trans': 'risk_dataframes/Reptile Trans - top20.csv',
        'FOMAML': 'risk_dataframes/FOMAML - top20.csv',
        'Balanced Bagging Classifier': 'risk_dataframes/risk_df_Balanced Bagging Classifier.csv'
    }
    ae = AdvancedEvaluator(plots_output_folder='advanced_ml_plots_top20/', models_results=models_results)
    ae.produce_empirical_risk_curves()
    ae.produce_roc_curves()
    ae.produce_curves_topK(topKs=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    models_results = {
        'MAML': 'risk_dataframes/MAML - top10.csv',
        # 'ARML-top20': 'arml/dm_top20/model_39/risk_df.csv',
        'Reptile': 'risk_dataframes/Reptile - top10.csv',
        'Reptile Trans': 'risk_dataframes/Reptile Trans - top10.csv',
        'FOMAML': 'risk_dataframes/FOMAML Trans - top10.csv',
        'Weighted Logistic Regression': 'risk_dataframes/risk_df_Weighted Logistic Regression.csv'
    }
    ae = AdvancedEvaluator(plots_output_folder='advanced_ml_plots_top10/', models_results=models_results)
    ae.produce_empirical_risk_curves()
    ae.produce_roc_curves()
    ae.produce_curves_topK(topKs=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])