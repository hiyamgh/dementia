import pandas as pd
import os


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def traverse_results(models, optimizer, output_folder, save_folder, name):
    encodings = ['catboost', 'glmm', 'target', 'mestimator', 'james', 'woe']
    path = os.path.join(output_folder, '{}/'.format(opt))
    sub_paths = [os.path.join(path, '{}/'.format(enc)) for enc in encodings]
    all_dfs = []
    for s in sub_paths:
        path1 = os.path.join(s, 'prob_results.csv')
        path2 = os.path.join(s, 'regular_results.csv')
        if os.path.isfile(path1):
            df = pd.read_csv(path1)
        else:
            df = pd.read_csv(path2)
        all_dfs.append(df)
    all_results = pd.concat(all_dfs).sort_values(by=optimizer, ascending=False).reset_index(drop=True)
    first_occurences = []
    for m in models:
        first_occurences.append(all_results[all_results.model_name == m].first_valid_index())
    all_results_cleaned = all_results[all_results.index.isin(first_occurences)].reset_index(drop=True)
    mkdir(save_folder)
    all_results_cleaned.to_csv(os.path.join(save_folder, name + '.csv'), index=False)


if __name__ == '__main__':
    optimizations = ['f2', 'gmean', 'bss', 'pr_auc', 'sensitivity', 'specificity']

    tops = [10, 20]
    all_models = {
        'probabilistic': ['Weighted XGBoost', 'Weighted Logistic Regression', 'Weighted Decision Tree Classifier',
                   'Balanced Bagging Classifier'],
        'regular': ['Weighted SVM', 'KNeighbors', 'Easy Ensemble Classifier']
    }

    for t in tops:
        for opt in optimizations:
            for m in all_models:
                if m == 'probabilistic' or (m == 'regular' and opt == 'f2'):
                    traverse_results(models=all_models[m], optimizer=opt,
                                     output_folder='../output/output_{}/top_{}/'.format(m, t),
                                     save_folder='../output/results/',
                                     name='top_{}_{}_{}'.format(t, m, opt))


