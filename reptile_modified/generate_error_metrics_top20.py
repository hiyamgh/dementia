import pandas as pd
import pickle
import numpy as np
import os
import string


def get_results_dementia(trained_models_dir, outfile_name, output_folder, sort_by, topn=3,):
    metrics = ['model', 'f2', 'gmean', 'bss', 'pr_auc', 'sensitivity', 'specificity', 'ppv',
               'accuracy', 'precision', 'recall', 'f1', 'auc']

    models = {}
    # count = 1
    for metric in sort_by:
        models[metric] = {}
        classification_results = pd.DataFrame(columns=metrics)
        for root, dirs, files in os.walk(trained_models_dir):
            if 'testing_error_metrics.p' in files:

                # get error metrics
                with open(os.path.join(root, 'testing_error_metrics.p'), 'rb') as f:
                    error_metrics = pickle.load(f)
                # get the risk df
                risk_df = pd.read_csv(os.path.join(root, 'risk_df.csv'))

                # get std of error metrics
                #with open(os.path.join(root, 'std_metrics.p'), 'rb') as f:
                #std_metrics = pickle.load(f)

                # print('root model: {}'.format(root))
                model_count = root[root.index('model_'):]
                # get the count of the model from the name of the model's folder
                # count = model_count.split("_")[1]

                models[metric][model_count] = {}
                models[metric][model_count]['error_metrics'] = error_metrics
                models[metric][model_count]['risk_df'] = risk_df

                # get the probability of the positive class,
                # only for instances that are marked highly probable
                # of issuing the dementia disease i.e. get the ppv
                # only when risk >= 0.9 for example
                risk_df_high = risk_df[risk_df['risk_scores'] >= 0.8]

                classification_results = classification_results.append({
                    'model': model_count,
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

    metrics = ['f2']
    out_folder = 'results_errors/'

    models2 = get_results_dementia(trained_models_dir='FOMAML_trans_trained_models/20/',
                         outfile_name='FOMAML_trans_20',
                         output_folder=out_folder,
                         sort_by=metrics)
