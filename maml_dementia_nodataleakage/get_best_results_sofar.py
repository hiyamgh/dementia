import pickle
import os


def get_results_dementia(trained_models_dir):
    best_f2 = 0.0
    best_model = ''
    for root, dirs, files in os.walk(trained_models_dir):
        if 'error_metrics.p' in files and 'std_metrics.p' in files:
            with open(os.path.join(root, 'error_metrics.p'), 'rb') as f:
                error_metrics = pickle.load(f)

            if float(error_metrics['f2']) > float(best_f2):
                best_f2 = error_metrics['f2']
                best_model = root.split('_')[4]

    print('best f2 so far: {} in model {}'.format(best_f2, best_model))


if __name__ == '__main__':
    dir1 = 'dementia_without_fp_top10/'
    dir2 = 'dementia_without_fp_top20/'
    dir3 = 'dementia_with_fp_top10/'
    dir4 = 'dementia_with_fp_top20/'

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
