import pandas as pd
import pickle

def get_winning_hyps(df_errors, hyper_parameters, outfile_name, fp=True):
    if fp:
        hyps = ['model', 'miter', 'mbs', 'mlr', 'ulr', 'dh', 'afn', 'nu', 'ifp', 'fp_supp']
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
                'fp_supp': params['fp_supp']
            }, ignore_index=True)
        df_hyper.to_csv('{}.csv'.format(outfile_name), index=False)
    else:
        hyps = ['model', 'miter', 'mbs', 'mlr', 'ulr', 'dh', 'afn', 'nu', 'ifp']
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
                'ifp': params['ifp']
            }, ignore_index=True)
        df_hyper.to_csv('{}.csv'.format(outfile_name), index=False)


if __name__ == '__main__':
    with open('hyperparameters_with_fp.p', 'rb') as handle:
        hyperparameters = pickle.load(handle)

    df_accuracy = pd.read_csv('fake_news_with_fp_accuracy.csv')
    df_precision = pd.read_csv('fake_news_with_fp_precision.csv')

    get_winning_hyps(df_errors=df_accuracy,
                     hyper_parameters=hyperparameters,
                     outfile_name='winning_with_fp_accuracy', fp=True)

    get_winning_hyps(df_errors=df_precision,
                     hyper_parameters=hyperparameters,
                     outfile_name='winning_with_fp_precision', fp=True)

    with open('hyperparameters_without_fp.p', 'rb') as handle:
        hyperparameters = pickle.load(handle)
    df_accuracy = pd.read_csv('fake_news_without_fp_accuracy.csv')
    df_precision = pd.read_csv('fake_news_without_fp_precision.csv')

    get_winning_hyps(df_errors=df_accuracy,
                     hyper_parameters=hyperparameters,
                     outfile_name='winning_without_fp_accuracy', fp=False)

    get_winning_hyps(df_errors=df_precision,
                     hyper_parameters=hyperparameters,
                     outfile_name='winning_without_fp_precision', fp=False)
