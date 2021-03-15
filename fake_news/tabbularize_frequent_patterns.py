import os, pickle
import pandas as pd
import spintax


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    erroneous_codebook = pd.read_csv('input/erroneous_codebook_legal_outliers_filtered.csv')
    pooled = pd.read_csv('../input/pooled_data.csv')
    fi = pd.read_csv('input/feature_importance_modified.csv')
    min_supps = [0.7, 0.8, 0.9]
    fps_folder = 'dementia_colsmeta_top10/'

    most_imp_features = list(fi['Feature'])[:20]
    ecf = erroneous_codebook[erroneous_codebook['COLUMN'].isin(most_imp_features)]
    ecf = ecf['COLUMN']
    ecf.to_csv('features_with_erroneous.csv', index=False)

    # with open(os.path.join(fps_folder, 'fps_dementia_0.9.pickle'), 'rb') as f:
    #     fps_top10 = pickle.load(f)
    #
    # with open(os.path.join(fps_folder, 'colsmeta_dementia_0.9.pickle'), 'rb') as f:
    #     cols_meta = pickle.load(f)
    #
    # percentiles = ['min', '25th', '50th', '75th', 'max']
    # meta_df = pd.DataFrame(columns=['feature', 'min', '25th', '50th', '75th', 'max'])
    # for col in cols_meta:
    #     meta_df = meta_df.append({
    #         'feature': col,
    #         'min': cols_meta[col]['min'] if 'min' in cols_meta[col] else '',
    #         '25th': cols_meta[col]['25th'] if '25th' in cols_meta[col] else '',
    #         '50th': cols_meta[col]['50th'] if '50th' in cols_meta[col] else '',
    #         '75th': cols_meta[col]['75th'] if '75th' in cols_meta[col] else '',
    #         'max': cols_meta[col]['max'] if 'max' in cols_meta[col] else '',
    #     }, ignore_index=True)
    #
    # df = pd.DataFrame(columns=['frequent pattern'])
    # df['frequent pattern'] = fps_top10
    # descriptions, values = [], []
    # for fp in fps_top10:
    #     dstr, vstr = '', ''
    #     for sub_fp in fp:
    #         # feature name & feature's row in the erroneous codebook
    #         col_name = sub_fp.split('<')[1]
    #         row = erroneous_codebook.loc[erroneous_codebook['COLUMN'] == col_name]
    #
    #         # description str
    #         dstr += '{}: '.format(col_name)
    #         dstr += spintax.spin(row['description'].values[0] + "\n")
    #
    #         # values str
    #         vstr += '{}: '.format(col_name)
    #         vstr += spintax.spin(', '.join(str(e) for e in sorted(pooled[col_name].dropna().unique())) + "\n")
    #
    #     descriptions.append(dstr)
    #     values.append(vstr)
    #
    # df['values'] = values
    # df['descriptions'] = descriptions
    #
    # out_folder = 'frequent_patterns/'
    # mkdir(out_folder)
    # df.to_csv(os.path.join(out_folder, 'fp_0.9.csv'), index=False)
    # meta_df.to_csv(os.path.join(out_folder, 'meta_df.csv'), index=False)



