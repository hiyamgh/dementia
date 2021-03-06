import pandas as pd
import pickle
import os

# 'f2': [support_f2] + query_total_f2s,
#         'gmean': [support_gmean] + query_total_gmeans,
#         'bss': [support_bss] + query_total_bsss,
#         'pr_auc': [support_pr_auc] + query_total_pr_aucs,
#         'sensitivity': [support_sensitivity] + query_total_sensitivities,
#         'specificity': [support_specificity] + query_total_specificities

def get_results_fake_news(trained_models_dir, outfile_name, sort_by):
    metrics = ['model', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'f2',
               'gmean', 'bss', 'pr_auc', 'sensitivity', 'specificity']

    classification_results = pd.DataFrame(columns=metrics)
    models = {}
    count = 1
    for metric in sort_by:
        for root, dirs, files in os.walk(trained_models_dir):
            if 'error_metrics.p' in files and 'std_metrics.p' in files:

                # get error metrics
                with open(os.path.join(root, 'error_metrics.p'), 'rb') as f:
                    error_metrics = pickle.load(f)

                # get std of error metrics
                #with open(os.path.join(root, 'std_metrics.p'), 'rb') as f:
                #std_metrics = pickle.load(f)

                print('root model: {}'.format(root))

                # get the count of the model from the name of the model's folder
                count = root.split('_')[4]

                models[count] = {}
                models[count]['error_metrics'] = error_metrics

                classification_results = classification_results.append({
                    'model': 'model_{}'.format(count),
                    'accuracy': '{}'.format(error_metrics['accuracy']),
                    'precision': '{}'.format(error_metrics['precision']),
                    'recall': '{}'.format(error_metrics['recall']),
                    'f1': '{}'.format(error_metrics['f1']),
                    'auc': '{}'.format(error_metrics['roc']),
                    'gmean': '{}'.format(error_metrics['gmean']),
                    'f2': '{}'.format(error_metrics['f2']),
                    'bss': '{}'.format(error_metrics['bss']),
                    'pr_auc': '{}'.format(error_metrics['pr_auc']),
                    'sensitivity': '{}'.format(error_metrics['sensitivity']),
                    'specificity': '{}'.format(error_metrics['specificity']),
                    # 'bss', 'pr_auc', 'sensitivity', 'specificity'
                }, ignore_index=True)

                # count += 1

        #with open('{}.p'.format(outfile_name), 'wb') as f:
        #pickle.dump(models, f, pickle.HIGHEST_PROTOCOL)

        classification_results = classification_results.sort_values(by=[metric], ascending=False)
        classification_results.to_csv('{}_{}.csv'.format(outfile_name, metric), index=False)
    return models


if __name__ == '__main__':
    get_results_fake_news(trained_models_dir='dementia_without_fp_top10/',
                                   outfile_name='dementia_without_fp_top10',
                                   sort_by=['f2', 'accuracy'])
    get_results_fake_news(trained_models_dir='dementia_without_fp_top20/',
                          outfile_name='dementia_without_fp_top20',
                          sort_by=['f2', 'accuracy'])

    print("\n====================================================================================\n")
    get_results_fake_news(trained_models_dir='dementia_with_fp_top10/',
                          outfile_name='dementia_with_fp_top10',
                          sort_by=['f2', 'accuracy'])
    get_results_fake_news(trained_models_dir='dementia_with_fp_top20/',
                          outfile_name='dementia_with_fp_top20',
                          sort_by=['f2', 'accuracy'])

    # models = get_results_dementia(trained_models_dir='dementia_relu_fp_3/')
    # AE = AdvancedEvaluator(models_results=models,
    #                        plots_output_folder='plots_deep_dementia/',
    #                        fp_growth_output_folder='plots_deep_dementia/')
    #
    # AE.produce_empirical_risk_curves()
    # AE.compute_jaccard_similarity(topKs=[10, 100, 200, 300, 400, 500, 600, 700])
    # AE.produce_roc_curves()
    # AE.produce_curves_topK(topKs=[10, 100, 200, 300, 400, 500, 600, 700])


