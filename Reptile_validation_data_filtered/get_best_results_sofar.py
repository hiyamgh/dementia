import pickle
import os


def get_results_dementia(trained_models_dir):
    if not os.path.exists(trained_models_dir):
        print('{} does not exist'.format(trained_models_dir))
        return
    best_f2 = 0.0
    best_sensitivity, best_specificity, auc = 0.0, 0.0, 0.0
    best_model = ''
    for root, dirs, files in os.walk(trained_models_dir):
        if 'testing_error_metrics.p' in files:
            with open(os.path.join(root, 'testing_error_metrics.p'), 'rb') as f:
                error_metrics = pickle.load(f)

            if float(error_metrics['f2']) > float(best_f2):
                best_f2 = error_metrics['f2']
                best_sensitivity = error_metrics['sensitivity']
                best_specificity = error_metrics['specificity']
                auc = error_metrics['roc']
                best_model = root

    print('best f2 so far: {} in model {}'.format(best_f2, best_model))
    print('best sensitivity so far: {} in model {}'.format(best_sensitivity, best_model))
    print('best specificity so far: {} in model {}'.format(best_specificity, best_model))
    print('best roc auc so far: {} in model {}'.format(auc, best_model))


if __name__ == '__main__':
    dir6 = 'FOML_trans_trained_models/'

    print('dir: {}'.format(dir6))
    get_results_dementia(dir6)

