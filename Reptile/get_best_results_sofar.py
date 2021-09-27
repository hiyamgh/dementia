import pickle
import os


def get_results_dementia(trained_models_dir):
    if not os.path.exists(trained_models_dir):
        print('{} does not exist'.format(trained_models_dir))
        return
    best_f2 = 0.0
    best_model = ''
    for root, dirs, files in os.walk(trained_models_dir):
        if 'testing_error_metrics.p' in files:
            with open(os.path.join(root, 'testing_error_metrics.p'), 'rb') as f:
                error_metrics = pickle.load(f)

            if float(error_metrics['f2']) > float(best_f2):
                best_f2 = error_metrics['f2']
                best_model = root

    print('best f2 so far: {} in model {}'.format(best_f2, best_model))


if __name__ == '__main__':
    dir1 = 'reptile_trained_models/10/'
    dir2 = 'reptile_trained_models/20/'
    dir3 = 'reptile_trans_trained_models/10/'
    dir4 = 'reptile_trans_trained_models/20/'
    dir5 = 'FOML_trans_trained_models/10/'
    dir6 = 'FOML_trans_trained_models/20/'

    print('dir: {}'.format(dir1))
    get_results_dementia(dir1)
    print('======================')
    print('dir2: {}'.format(dir2))
    get_results_dementia(dir2)
    print('======================')
    print('dir3: {}'.format(dir3))
    get_results_dementia(dir3)
    print('======================')
    print('dir4: {}'.format(dir4))
    get_results_dementia(dir4)
    print('======================')
    print('dir5: {}'.format(dir5))
    get_results_dementia(dir5)
    print('======================')
    print('dir6: {}'.format(dir6))
    get_results_dementia(dir6)
    print('======================')


