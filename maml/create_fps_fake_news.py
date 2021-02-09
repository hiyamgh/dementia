import pandas as pd
import os
from helper import *
import pickle

if __name__ == '__main__':
    df_train = pd.read_csv('feature_extraction_train_updated.csv')
    df_test = pd.read_csv('feature_extraction_test_updated.csv')
    df = pd.concat([df_train, df_test])
    df = df.drop(['article_title', 'article_content', 'source', 'source_category', 'unit_id',
                  'label'],
                 axis=1)
    df_cols = list(df.columns)
    col_types = df.dtypes # Series object, access by column name # int64 or float64 [col_name] == np.float64

    min_supps = [0.5, 0.6, 0.7]
    out_folder = 'fake_news_fps/'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for supp in min_supps:
        itemSetList = []
        print('len of df: {}'.format(len(df)))
        for index, row in df.iterrows():
            curr_items = []
            for col in df_cols:
                if col_types[col] == np.float64:
                    val = float(row[col])
                else:
                    val = int(row[col])
                curr_items.append('{}={}'.format(col, val))

            itemSetList.append(curr_items)
        if supp == 0.8:
            print()
        freqItemSet, rules = fpgrowth(itemSetList, minSupRatio=supp, minConf=supp)
        if freqItemSet:
            print('Frequent patterns: ')
            for fp in freqItemSet:
                print(fp)

        with open(os.path.join(out_folder, 'fps_fakenews_{}.pickle'.format(supp)), 'wb') as handle:
            pickle.dump(freqItemSet, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('dumped {} into {}'.format('fps_fakenews_{}.pickle'.format(supp), out_folder))
