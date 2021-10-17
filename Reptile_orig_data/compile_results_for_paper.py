import pandas as pd
import pickle
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', default=10, help='top 10 or top 20 columns results')
    parser.add_argument('--num', default=30, help='number of best models')
    parser.add_argument('--type', default='reptile')
    args = parser.parse_args()

    if args.type == 'reptile':
        df_results = pd.read_csv('results_errors/reptile_{}_f2.csv'.format(args.top))
    elif args.type == 'reptile_trans':
        df_results = pd.read_csv('results_errors/reptile_trans_{}_f2.csv'.format(args.top))
    else:
        df_results = pd.read_csv('results_errors/foml_trans_{}_f2.csv'.format(args.top))
    best_models = []
    for idx, row in df_results.iterrows():
        if idx <= int(args.num):
            best_model = int(row['model'].split('_')[1])
            best_models.append(best_model)
        else:
            break

    df_final_results = pd.DataFrame(columns=['model', 'f2', 'gmean', 'bss', 'pr_auc', 'sensitivity', 'specificity',
                                            'accuracy', 'precision', 'recall', 'f1', 'auc'])

    df_final_hyps = pd.DataFrame(columns=['model', 'shots', 'inner_batch', 'inner_iters', 'lr',
                                          'meta_step', 'meta_step_final', 'meta_batch', 'meta_iters',
                                          'eval_batch', 'eval_iters', 'dim_hidden', 'activation_fns',
                                          'top_features'])

    with open('idx2hyps.pkl', 'rb') as handle:
        hyps = pickle.load(handle)

    for bm in best_models:
        # bm: best model (count) -- it must be in the dictionary mapping indexes (count) to hyperparameters
        if bm in hyps:
            # save winning hyperparameters
            df_final_hyps = df_final_hyps.append({
                'model': str(bm),
                'shots':           hyps[bm]['shots'],
                'inner_batch':     hyps[bm]['inner_batch'],
                'inner_iters':     hyps[bm]['inner_iters'],
                'lr':              hyps[bm]['lr'],
                'meta_step':       hyps[bm]['meta_step'],
                'meta_step_final': hyps[bm]['meta_step_final'],
                'meta_batch':      hyps[bm]['meta_batch'],
                'meta_iters':      hyps[bm]['meta_iters'],
                'eval_batch':      hyps[bm]['eval_batch'],
                'eval_iters':      hyps[bm]['eval_iters'],
                'dim_hidden':      hyps[bm]['dim_hidden'],
                'activation_fns':  hyps[bm]['activation_fns'],
                'top_features':    hyps[bm]['top_features']
            }, ignore_index=True)

            exp_string = 'FOML_trans_trained_models/{}/model_{}/'.format(args.top, bm)

            # save dictionary of results
            with open(os.path.join(exp_string, 'testing_error_metrics.p'), 'rb') as f:
                test_results = pickle.load(f)
            # save dictionary of results
            with open(os.path.join(exp_string, 'training_error_metrics.p'), 'rb') as f:
                train_results = pickle.load(f)

            # save testing errors
            df_final_results = df_final_results.append({
                'model': str(bm) + '_test',
                'gmean': '{}'.format(test_results['gmean']),
                'f2': '{}'.format(test_results['f2']),
                'bss': '{}'.format(test_results['bss']),
                'pr_auc': '{}'.format(test_results['pr_auc']),
                'sensitivity': '{}'.format(test_results['sensitivity']),
                'specificity': '{}'.format(test_results['specificity']),
                'accuracy': '{}'.format(test_results['accuracy']),
                'precision': '{}'.format(test_results['precision']),
                'recall': '{}'.format(test_results['recall']),
                'f1': '{}'.format(test_results['f1']),
                'auc': '{}'.format(test_results['roc']),
                # 'bss', 'pr_auc', 'sensitivity', 'specificity'
            }, ignore_index=True)

            # save training errors
            df_final_results = df_final_results.append({
                'model': str(bm) + '_train',
                'gmean': '{}'.format(train_results['gmean']),
                'f2': '{}'.format(train_results['f2']),
                'bss': '{}'.format(train_results['bss']),
                'pr_auc': '{}'.format(train_results['pr_auc']),
                'sensitivity': '{}'.format(train_results['sensitivity']),
                'specificity': '{}'.format(train_results['specificity']),
                'accuracy': '{}'.format(train_results['accuracy']),
                'precision': '{}'.format(train_results['precision']),
                'recall': '{}'.format(train_results['recall']),
                'f1': '{}'.format(train_results['f1']),
                'auc': '{}'.format(train_results['roc']),
                # 'bss', 'pr_auc', 'sensitivity', 'specificity'
            }, ignore_index=True)

            # make an empty row after displaying training and testing errors
            df_final_results = df_final_results.append({
                'model': '',
                'gmean': '',
                'f2': '',
                'bss': '',
                'pr_auc': '',
                'sensitivity': '',
                'specificity': '',
                'accuracy': '',
                'precision': '',
                'recall': '',
                'f1': '',
                'auc': '',
                # 'bss', 'pr_auc', 'sensitivity', 'specificity'
            }, ignore_index=True)
        else:
            continue

    df_final_hyps.to_csv('results_errors/final_hyps_top{}.csv'.format(args.top), index=False)
    df_final_results.to_csv('results_errors/final_results_top{}.csv'.format(args.top), index=False)