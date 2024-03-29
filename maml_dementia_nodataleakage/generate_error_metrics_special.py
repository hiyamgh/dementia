import pandas as pd
import pickle
import numpy as np
import os
from AdvancedEvaluationDL import AdvancedEvaluator


def get_results_dementia(trained_models_dir, outfile_name, output_folder, sort_by, topn=3,):
    metrics = ['model', 'f2', 'gmean', 'bss', 'pr_auc', 'sensitivity', 'specificity', 'ppv',
               'accuracy', 'precision', 'recall', 'f1', 'auc']

    models = {}
    # count = 1
    for metric in sort_by:
        models[metric] = {}
        classification_results = pd.DataFrame(columns=metrics)
        for root, dirs, files in os.walk(trained_models_dir):
            if 'error_metrics.p' in files and 'std_metrics.p' in files:

                # get error metrics
                with open(os.path.join(root, 'error_metrics.p'), 'rb') as f:
                    error_metrics = pickle.load(f)
                # get the risk df
                risk_df = pd.read_csv(os.path.join(root, 'risk_df.csv'))

                # get std of error metrics
                #with open(os.path.join(root, 'std_metrics.p'), 'rb') as f:
                #std_metrics = pickle.load(f)

                # print('root model: {}'.format(root))

                # get the count of the model from the name of the model's folder
                count = root.split('_')[4]

                models[metric]['model_{}'.format(count)] = {}
                models[metric]['model_{}'.format(count)]['error_metrics'] = error_metrics
                models[metric]['model_{}'.format(count)]['risk_df'] = risk_df

                # get the probability of the positive class,
                # only for instances that are marked highly probable
                # of issuing the dementia disease i.e. get the ppv
                # only when risk >= 0.9 for example
                risk_df_high = risk_df[risk_df['risk_scores'] >= 0.8]

                classification_results = classification_results.append({
                    'model': 'model_{}'.format(count),
                    'gmean': '{}'.format(error_metrics['gmean']),
                    'f2': '{}'.format(error_metrics['f2']),
                    'bss': '{}'.format(error_metrics['bss']),
                    'pr_auc': '{}'.format(error_metrics['pr_auc']),
                    'sensitivity': '{}'.format(error_metrics['sensitivity']),
                    'specificity': '{}'.format(error_metrics['specificity']),
                    'ppv': '{}'.format(np.mean(risk_df_high['risk_scores'])),
                    'accuracy': '{}'.format(error_metrics['accuracy']),
                    'precision': '{}'.format(error_metrics['precision']),
                    'recall': '{}'.format(error_metrics['recall']),
                    'f1': '{}'.format(error_metrics['f1']),
                    'auc': '{}'.format(error_metrics['roc']),
                    # 'bss', 'pr_auc', 'sensitivity', 'specificity'
                }, ignore_index=True)

                # count += 1

        #with open('{}.p'.format(outfile_name), 'wb') as f:
        #pickle.dump(models, f, pickle.HIGHEST_PROTOCOL)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        classification_results = classification_results.sort_values(by=[metric], ascending=False)

        top_models = list(classification_results['model'])[:topn]
        models[metric]['topn_models'] = top_models
        print('top {} models for {} in {}: {}'.format(topn, metric, outfile_name, top_models))

        classification_results.to_csv(os.path.join(output_folder, '{}_{}.csv'.format(outfile_name, metric)), index=False)

    return models


if __name__ == '__main__':

    metrics = ['f2', 'bss']
    out_folder = 'results_errors_special/'

    models1 = get_results_dementia(trained_models_dir='dementia_without_fp_top10/',
                         outfile_name='dementia_without_fp_top10',
                         output_folder=out_folder,
                         sort_by=metrics)

    all_models = [models1]
    all_models_names = ['without_top10']
    all_tuples = list(zip(all_models, all_models_names))
    special_models = ['model_350', 'model_353', 'model_278', 'model_269']
    for models, models_names in all_tuples:
        AE = AdvancedEvaluator(models_results=models,
                               plots_output_folder='advanced_ml_plots/{}_special/'.format(models_names),
                               special_cases=special_models)

        AE.produce_empirical_risk_curves()
        AE.compute_jaccard_similarity(topKs=[10, 100, 200, 300, 400, 500, 600, 700])
        AE.produce_roc_curves()
        AE.produce_curves_topK(topKs=[10, 100, 200, 300, 400, 500, 600, 700])