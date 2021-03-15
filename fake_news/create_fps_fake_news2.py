import pandas as pd
import os
from helper import *
import pickle


def identify_frequent_patterns(df, target_variable, supp_fp):
    # inputs needed
    itemSetList = []
    df_cols = list(df.columns)
    df_cols.remove(target_variable)

    #  get the 25th, 50th, and 75th quartiles of each column
    cols_meta = {}

    for col in df_cols:
        cols_meta[col] = {
            'min': df[col].min(),
            '25th': df[col].quantile(0.25),
            '50th': df[col].quantile(0.50),
            '75th': df[col].quantile(0.75),
            'max': df[col].max()
        }

        keys = list(cols_meta[col].keys())
        values = list(cols_meta[col].values())
        keys_to_delete = []
        for i in range(len(values)-1):
            if values[i] == values[i+1]:
                keys_to_delete.append(keys[i])

        if keys_to_delete:
            for k in keys_to_delete:
                del cols_meta[col][k]

    # use these quantiles for categorizing data
    for index, row in df.iterrows():
        curr_items = []
        for col in df_cols:
            percentiles = list(cols_meta[col].keys())
            percentiles_pairs = list(zip(percentiles, percentiles[1:]))
            for pair in percentiles_pairs:
                if pair[1] != 'max':
                    if cols_meta[col][pair[0]] <= row[col] < cols_meta[col][pair[1]]:
                        curr_items.append('{}<{}<{}'.format(cols_meta[col][pair[0]], col, cols_meta[col][pair[1]]))
                        break
                else:
                    curr_items.append(
                        '{}<{}<{}'.format(cols_meta[col][pair[0]], col, cols_meta[col][pair[1]]))

        itemSetList.append(curr_items)

    # get the frequent patterns -- list of sets
    freqItemSet, rules = fpgrowth(itemSetList, minSupRatio=supp_fp, minConf=supp_fp)
    if freqItemSet:
        print('Frequent patterns: ')
        for fp in freqItemSet:
            print(fp)
    return freqItemSet, cols_meta


if __name__ == '__main__':
    df_train = pd.read_csv('feature_extraction_train_updated.csv')
    df_test = pd.read_csv('feature_extraction_test_updated.csv')
    df = pd.concat([df_train, df_test])
    df = df.drop(['article_title', 'article_content', 'source', 'source_category', 'unit_id'],axis=1)
    df_cols = list(df.columns)
    col_types = df.dtypes # Series object,
    # access by column name # int64 or float64 [col_name] == np.float64

    min_supps = [0.5, 0.6, 0.7, 0.8, 0.9]

    out_folder = 'fake_news_fps_colsmeta/'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for supp in min_supps:
        print('\n======================================== support: {} ========================================'.format(supp))
        freqItemSet, cols_meta = identify_frequent_patterns(df, target_variable='label', supp_fp=supp)
        with open(os.path.join(out_folder, 'fps_fakenews_{}.pickle'.format(supp)), 'wb') as handle:
            pickle.dump(freqItemSet, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(out_folder, 'colsmeta_fakenews_{}.pickle'.format(supp)), 'wb') as handle:
            pickle.dump(cols_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)