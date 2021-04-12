import pandas as pd
import os


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def concat_results(df_meta, df_shallow, metric, output_folder, file_name):
    df_meta = df_meta.sort_values(by=metric, ascending=False)
    df_shallow = df_shallow.sort_values(by=metric, ascending=False)

    df_concatenated = pd.concat([df_meta.head(3), df_shallow.head(3)], sort='True')
    df_concatenated = df_concatenated.sort_values(by=metric, ascending=False)

    mkdir(output_folder)

    df_concatenated = df_concatenated.sort_values(by=metric, ascending=False)
    df_concatenated = df_concatenated[['model'] + ['f2', 'gmean', 'bss', 'pr_auc']]
    df_concatenated.to_csv(os.path.join(output_folder, file_name + '.csv'), index=False)

    return df_concatenated


if __name__ == '__main__':
    # top 10 with top 10
    metrics = ['f2', 'gmean', 'bss', 'pr_auc']

    results_meta_top10 = pd.read_csv('C:/Users/96171/Downloads/todaysresults/results_errors/results_errors/dementia_with_fp_top10_f2.csv')
    results_shallow_top10_prob = pd.read_csv('../output/output_probabilistic/top10/prob_results.csv')
    results_shallow_top10_prob.rename(columns={'model_name': 'model'}, inplace=True)

    results_meta_top10 = results_meta_top10[metrics + ['model']]
    results_shallow_top10_prob = results_shallow_top10_prob[metrics + ['model']]

    for met in metrics:
        concat_results(df_meta=results_meta_top10, df_shallow=results_shallow_top10_prob,
                       metric=met, output_folder='meta_vs_shallow/', file_name='top10_{}'.format(met))

    ###################################################################################################
    results_meta_top20 = pd.read_csv('C:/Users/96171/Downloads/todaysresults/results_errors/results_errors/dementia_with_fp_top20_f2.csv')
    results_shallow_top20_prob = pd.read_csv('../output/output_probabilistic/top20/prob_results.csv')
    results_shallow_top20_prob.rename(columns={'model_name': 'model'}, inplace=True)

    results_meta_top20 = results_meta_top20[metrics + ['model']]
    results_shallow_top20_prob = results_shallow_top20_prob[metrics + ['model']]

    for met in metrics:
        concat_results(df_meta=results_meta_top20, df_shallow=results_shallow_top20_prob,
                       metric=met, output_folder='meta_vs_shallow/', file_name='top20_{}'.format(met))
