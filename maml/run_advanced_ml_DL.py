import pandas as pd
import pickle
import os
from AdvancedEvaluationDL import AdvancedEvaluator


def get_results(trained_models_dir):
    classification_results = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall',
                                                            'f1', 'auc'])
    if not os.path.isfile('all_results.p'):
        models = {}
        count = 1
        countall = 1
        for root, dirs, files in os.walk(trained_models_dir):
            if 'error_metrics.p' in files:
                with open(os.path.join(root, 'error_metrics.p'), 'rb') as f:
                    error_metrics = pickle.load(f)
                if float(error_metrics['accuracy'].split('+-')[0]) >= 0.8:
                    print('root model: {}'.format(root))
                    models[count] = {}
                    risk_df = pd.read_csv(os.path.join(root, 'risk_df.csv'))
                    models[count]['risk_df'] = risk_df
                    models[count]['error_metrics'] = error_metrics
                    count += 1

                countall += 1

                classification_results = classification_results.append({
                    'model': 'model{}'.format(countall),
                    'accuracy': '{}'.format(error_metrics['accuracy']),
                    'precision': '{}'.format(error_metrics['precision']),
                    'recall': '{}'.format(error_metrics['recall']),
                    'f1': '{}'.format(error_metrics['f1']),
                    'auc': '{}'.format(error_metrics['roc'])
                }, ignore_index=True)

        with open('all_results.p', 'wb') as f:
            pickle.dump(models, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('all_results.p', 'rb') as f:
            models = pickle.load(f)

    classification_results = classification_results.sort_values(by=['accuracy'], ascending=False)
    classification_results.to_csv('error_metrics_deep.csv', index=False)
    return models


if __name__ == '__main__':
    models = get_results(trained_models_dir='jobs/')
    AE = AdvancedEvaluator(models_results=models,
                           plots_output_folder='plots_deep/',
                           fp_growth_output_folder='plots_deep/')

    AE.produce_empirical_risk_curves()
    AE.compute_jaccard_similarity(topKs=[10, 100, 200, 300, 400, 500, 600, 700])
    AE.produce_roc_curves()
    AE.produce_curves_topK(topKs=[10, 100, 200, 300, 400, 500, 600, 700])


