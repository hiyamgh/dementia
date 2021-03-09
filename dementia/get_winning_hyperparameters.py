import pandas as pd
import pickle
import os


def get_winning_hyps(df_errors, hyper_parameters, outfile_name, output_folder, fp=True):
    if fp:
        hyps = ['model', 'miter', 'mbs', 'mlr', 'ulr', 'dh', 'afn', 'nu', 'ifp', 'fp_supp', 'weights', 'sampling_strategy']
        df_hyper = pd.DataFrame(columns=hyps)
        for model_name in list(df_errors['model']):
            params = hyper_parameters[model_name]
            df_hyper = df_hyper.append({
                'model': model_name,
                'miter': params['miter'],
                'mbs': params['mbs'],
                'mlr': params['mlr'],
                'ulr': params['ulr'],
                'dh': params['dh'],
                'afn': params['afn'],
                'nu': params['nu'],
                'ifp': params['ifp'],
                'fp_supp': params['fp_supp'],
                'weights': params['weights'],
                'sampling_strategy': params['sampling_strategy']
            }, ignore_index=True)
        df_hyper.to_csv('{}.csv'.format(outfile_name), index=False)
    else:
        hyps = ['model', 'miter', 'mbs', 'mlr', 'ulr', 'dh', 'afn', 'nu', 'ifp', 'weights', 'sampling_strategy']
        df_hyper = pd.DataFrame(columns=hyps)
        for model_name in list(df_errors['model']):
            params = hyper_parameters[model_name]
            df_hyper = df_hyper.append({
                'model': model_name,
                'miter': params['miter'],
                'mbs': params['mbs'],
                'mlr': params['mlr'],
                'ulr': params['ulr'],
                'dh': params['dh'],
                'afn': params['afn'],
                'nu': params['nu'],
                'ifp': params['ifp'],
                'weights': params['weights'],
                'sampling_strategy': params['sampling_strategy']
            }, ignore_index=True)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df_hyper.to_csv(os.path.join(output_folder, '{}.csv'.format(outfile_name)), index=False)


if __name__ == '__main__':
    with open('hyperparameters_with_fp.p', 'rb') as handle:
        hyperparameters = pickle.load(handle)

    output_folder = 'results_hyperparameters/'

    df_top10_f2 = pd.read_csv('results_errors/dementia_with_fp_top10_f2.csv')
    df_top10_bss = pd.read_csv('results_errors/dementia_with_fp_top10_bss.csv')

    df_top20_f2 = pd.read_csv('results_errors/dementia_with_fp_top20_f2.csv')
    df_top20_bss = pd.read_csv('results_errors/dementia_with_fp_top20_bss.csv')

    get_winning_hyps(df_errors=df_top10_f2,
                     hyper_parameters=hyperparameters,
                     outfile_name='winning_top10_with_fp_f2',
                     output_folder=output_folder,
                     fp=True)

    get_winning_hyps(df_errors=df_top10_bss,
                     hyper_parameters=hyperparameters,
                     outfile_name='winning_top10_with_fp_bss',
                     output_folder=output_folder,
                     fp=True)

    get_winning_hyps(df_errors=df_top20_f2,
                     hyper_parameters=hyperparameters,
                     outfile_name='winning_top20_with_fp_f2',
                     output_folder=output_folder,
                     fp=True)

    get_winning_hyps(df_errors=df_top20_bss,
                     hyper_parameters=hyperparameters,
                     outfile_name='winning_top20_with_fp_bss',
                     output_folder=output_folder,
                     fp=True)

    with open('hyperparameters_without_fp.p', 'rb') as handle:
        hyperparameters_without = pickle.load(handle)

    df_top10_f2_without = pd.read_csv('results_errors/dementia_without_fp_top10_f2.csv')
    df_top10_bss_without = pd.read_csv('results_errors/dementia_without_fp_top10_bss.csv')

    df_top20_f2_without = pd.read_csv('results_errors/dementia_without_fp_top20_f2.csv')
    df_top20_bss_without = pd.read_csv('results_errors/dementia_without_fp_top20_bss.csv')

    get_winning_hyps(df_errors=df_top10_f2_without,
                     hyper_parameters=hyperparameters_without,
                     outfile_name='winning_top10_without_fp_f2',
                     output_folder=output_folder,
                     fp=False)

    get_winning_hyps(df_errors=df_top10_bss_without,
                     hyper_parameters=hyperparameters_without,
                     outfile_name='winning_top10_without_fp_bss',
                     output_folder=output_folder,
                     fp=False)

    get_winning_hyps(df_errors=df_top20_f2_without,
                     hyper_parameters=hyperparameters_without,
                     outfile_name='winning_top20_without_fp_f2',
                     output_folder=output_folder,
                     fp=False)

    get_winning_hyps(df_errors=df_top20_bss_without,
                     hyper_parameters=hyperparameters_without,
                     outfile_name='winning_top20_without_fp_bss',
                     output_folder=output_folder,
                     fp=False)
