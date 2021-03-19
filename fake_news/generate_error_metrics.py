import pandas as pd
import pickle
import numpy as np
import os
from AdvancedMLEvaluationDL import AdvancedEvaluator
from get_hyper_parameters import *


def get_results_dementia(trained_models_dir, outfile_name, output_folder, sort_by, topn=3,):
    metrics = ['model', 'accuracy', 'precision', 'recall', 'f1', 'auc']

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
                #risk_df_high = risk_df[risk_df['risk_scores'] >= 0.8]

                classification_results = classification_results.append({
                    'model': 'model_{}'.format(count),
                    'accuracy': '{}'.format(error_metrics['accuracy']),
                    'precision': '{}'.format(error_metrics['precision']),
                    'recall': '{}'.format(error_metrics['recall']),
                    'f1': '{}'.format(error_metrics['f1']),
                    'auc': '{}'.format(error_metrics['roc']),
                }, ignore_index=True)

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

    metrics = ['accuracy']
    out_folder = 'results_errors/'
    # df_train = pd.read_csv('input/feature_extraction_train_updated.csv')
    df_test = pd.read_csv('input/feature_extraction_test_updated.csv')

    models1 = get_results_dementia(trained_models_dir='fake_news_with_fp2/',
                         outfile_name='fake_news_with_fp',
                         output_folder=out_folder,
                         sort_by=metrics)

    models2 = get_results_dementia(trained_models_dir='fake_news_without_fp/',
                                   outfile_name='fake_news_without_fp',
                                   output_folder=out_folder,
                                   sort_by=metrics)
    
    all_models = [models1, models2]
    all_models_names = ['with_fp', 'without_fp']

    all_tuples = list(zip(all_models, all_models_names))
    for models, models_names in all_tuples:
        # get the fp min supp that the model used
        if models_names == 'with_fp':

            # this will save hyper parameters to pickle file
            get_hyper_parameters(outfile_name='hyperparameters_with_fp', fp=True)
            # read the pickle file of hyper parameters
            with open('hyperparameters_with_fp.p', 'rb') as handle:
                hyperparameters = pickle.load(handle)

            out_folder = 'advanced_ml_plots/{}/'.format(models_names)
            AE = AdvancedEvaluator(models_results=models,
                                   plots_output_folder=out_folder,
                                   hyperparameters=hyperparameters,
                                   dir_of_fps='fake_news_fps_colsmeta/',
                                   suffix='fps_fakenews',
                                   df_test=df_test)

            AE.produce_empirical_risk_curves()
            AE.compute_jaccard_similarity(topKs=[10, 100, 200, 300, 400, 500, 600, 700])
            AE.produce_roc_curves()
            AE.produce_curves_topK(topKs=[10, 100, 200, 300, 400, 500, 600, 700])
            AE.characterize_prediction_mistakes(out_folder=out_folder)

            break
