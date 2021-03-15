import os, pickle
import pandas as pd


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def create_colsmeta_df(cols_meta):
    meta_df = pd.DataFrame(columns=['feature', 'min', '25th', '50th', '75th', 'max'])
    for col in cols_meta:
        meta_df = meta_df.append({
            'feature': col,
            'min': cols_meta[col]['min'] if 'min' in cols_meta[col] else '',
            '25th': cols_meta[col]['25th'] if '25th' in cols_meta[col] else '',
            '50th': cols_meta[col]['50th'] if '50th' in cols_meta[col] else '',
            '75th': cols_meta[col]['75th'] if '75th' in cols_meta[col] else '',
            'max': cols_meta[col]['max'] if 'max' in cols_meta[col] else '',
        }, ignore_index=True)

    return meta_df


if __name__ == '__main__':
    df_train = pd.read_csv('input/feature_extraction_train_updated.csv', encoding='latin-1')
    df_test = pd.read_csv('input/feature_extraction_test_updated.csv', encoding='latin-1')
    df = pd.concat([df_train, df_test])
    fps_folder = 'fake_news_fps_colsmeta/'

    df = df.drop(['article_title', 'article_content', 'source', 'source_category', 'unit_id', 'label'], axis=1)

    # get the cols_meta pickle files
    with open(os.path.join(fps_folder, 'colsmeta_fakenews_0.7.pickle'), 'rb') as f:
        cols_meta = pickle.load(f)

    # creating data frame for cols meta
    meta_df = create_colsmeta_df(cols_meta) # cols meta dataset

    out_folder = 'frequent_patterns/'
    mkdir(out_folder)

    df.describe().to_csv(os.path.join(out_folder, 'desc_df.csv')) # description dataset
    meta_df.to_csv(os.path.join(out_folder, 'meta_df.csv'), index=False) # colsmeta dataset

    minn_supps = [0.5, 0.6, 0.7, 0.8, 0.9]
    for supp in minn_supps:
        with open(os.path.join(fps_folder, 'fps_fakenews_{}.pickle'.format(supp)), 'rb') as f:
            fps = pickle.load(f)

        # creating dataset for the frequent patterns
        fp_df = pd.DataFrame()
        fp_df['frequent_patterns'] = fps

        # save output datasets
        fp_df.to_csv(os.path.join(out_folder, 'fp_{}.csv'.format(supp)), index=False)



